"""Стадия train: LoRA-дообучение на выходе стадии tokenize.

    python -m src.train --variant all_layers          # полный прогон варианта
    python -m src.train --variant all_layers --max-steps 4 --out /tmp/x   # smoke

Цикл обучения написан руками, а не через Trainer: так видно всё, что
обычно прячется, — где считается val loss, как копятся градиенты, что
сохраняется рядом с адаптером.

Что сохраняется в models/adapter_<variant>/: адаптер и токенизатор с chat template.
В metrics/train_<variant>.json — кривые train/val loss, время, пиковая память,
число обучаемых параметров, вес адаптера и отпечаток входов.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Потолок памяти Metal — до импорта torch. Без него mps занимает сколько дадут,
# и на ноутбуке с 16–32 ГБ система уходит в своп вместо внятной ошибки.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")   # нижний порог не выше верхнего

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, batches, load_split
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed, rng_state, restore_rng


TRAIN_CODE = ("src/train.py", "src/data.py", "src/runtime.py", "src/config.py")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants")


def inputs_fingerprint(params: dict) -> str:
    """Отпечаток кода обучения и секций конфига, от которых зависит адаптер.

    check.sh сверяет его с текущим: правили train.py или lr, а адаптер
    остался от прошлого прогона — проверять его бессмысленно.
    """
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    h.update(Path("uv.lock").read_bytes())
    for split in ("train", "val"):
        with Path(params["data"][split]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    cfg = params["lora"]
    if not 0 <= freeze_first < n_layers:
        raise ValueError(f"freeze_first={freeze_first}, слоёв {n_layers}")
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        layers_to_transform=list(range(freeze_first, n_layers)),
        task_type="CAUSAL_LM",
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int) -> float:
    """Средний лосс на токен по всему val-сплиту.

    Среднее по батчам нельзя: в батчах разное число токенов под маской.
    Поэтому сумма лоссов, взвешенная числом токенов, делённая на их сумму.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
            batch = {k: v.to(device) for k, v in batch.items()}
            n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
            if n == 0:
                continue
            loss = model(**batch).loss
            total += loss.item() * n
            count += n
    finally:
        model.train(was_training)
        if device.type == "mps":
            torch.mps.empty_cache()
    if not count:
        raise ValueError("val не содержит токенов ответа")
    return total / count


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="куда писать адаптер и метрики (smoke-тесты)")
    ap.add_argument("--val-limit", type=int, default=None, help="оценивать на первых N примерах val (smoke)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="сохранить контрольную точку и выйти на заданном шаге (проверка восстановления)")
    args = ap.parse_args()

    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}, есть: {sorted(variants)}")
    variant = variants[args.variant]
    tcfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")

    set_seed(tcfg["seed"])  # до загрузки базы, инициализации LoRA и dropout
    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])

    train_blob = load_split(params["data"]["train"])
    val_blob = load_split(params["data"]["val"])
    if args.val_limit:
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
    for blob in (train_blob, val_blob):
        if blob.get("model") != params["model"]["name"]:
            raise ValueError("Модель в тензорах ДЗ4 отличается от model.name")
    pad_id = train_blob["pad_token_id"]
    if val_blob["pad_token_id"] != pad_id:
        raise ValueError("У train и val разные pad_token_id")

    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"])
    if tokenizer.pad_token_id != pad_id:
        raise ValueError("pad_token_id токенизатора отличается от тензоров ДЗ4")
    tokenizer.padding_side = train_blob.get("padding_side", "left")
    model = AutoModelForCausalLM.from_pretrained(params["model"]["name"], dtype=dtype).to(device)
    n_layers = model.config.num_hidden_layers
    if params["train"].get("gradient_checkpointing"):
        # Активации 28 слоёв не храним, а пересчитываем на обратном проходе:
        # памяти в разы меньше, шаг примерно на треть дольше.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, n_layers, variant["freeze_first"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    examples = train_blob["examples"]
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps должен быть положительным")
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps
    )

    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    checkpoint_path = out_root / f"checkpoint_{args.variant}.pt"
    fingerprint = inputs_fingerprint(params)
    run_identity = {"inputs": fingerprint, "variant": args.variant,
                    "steps": total_steps, "val_limit": args.val_limit, "device": device.type}
    saved = None
    if checkpoint_path.exists():
        # Только собственная локальная контрольная точка, никогда внешний адаптер.
        candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if candidate["identity"] == run_identity:
            saved = candidate
        else:
            print("Контрольная точка другого эксперимента — начинаю новый", flush=True)

    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    if saved is None:
        print(f"[{args.variant}] оценка базы на {len(val_blob['examples'])} примерах", flush=True)
        eval_started = time.perf_counter()
        base_val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs)
        base_eval_seconds = time.perf_counter() - eval_started
        curve_train, curve_val = [], [[0, round(base_val, 4)]]
        step, micro, seen_tokens, seen_examples = 0, 0, 0, 0
        first_epoch, skip_micro, resumes = 0, 0, 0
        previous_seconds, eval_seconds = 0.0, 0.0
        peak = allocated_bytes(device)
        print(f"[{args.variant}] шаг 0: val {base_val:.4f}", flush=True)
    else:
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], device)
        base_val, base_eval_seconds = saved["base_val"], saved["base_eval_seconds"]
        curve_train, curve_val = saved["curve_train"], saved["curve_val"]
        step, micro = saved["step"], saved["micro"]
        seen_tokens, seen_examples = saved["seen_tokens"], saved["seen_examples"]
        first_epoch, skip_micro = saved["next_epoch"], saved["skip_micro"]
        previous_seconds, eval_seconds = saved["seconds"], saved["eval_seconds"]
        peak, resumes = max(saved["peak"], allocated_bytes(device)), saved["resumes"] + 1
        print(f"[{args.variant}] продолжение с шага {step}/{total_steps}", flush=True)
        del candidate, saved  # CPU-копии не должны влиять на расход памяти прогона
    print(f"[{args.variant}] устройство {device}, обучаемых {trainable:,} из {total:,} "
          f"({trainable / total:.3%}); шагов {total_steps}", flush=True)

    initial_eval_seconds = eval_seconds
    started = time.perf_counter()
    accum_loss, accum_tokens, diverged = 0.0, 0, False

    def elapsed_train():
        return previous_seconds + time.perf_counter() - started - (eval_seconds - initial_eval_seconds)

    def checkpoint(epoch: int, micro_idx: int) -> None:
        state = {
            "identity": run_identity,
            "adapter": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(
                model, save_embedding_layers=False).items()},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(device), "base_val": base_val,
            "base_eval_seconds": base_eval_seconds, "curve_train": curve_train,
            "curve_val": curve_val, "step": step, "micro": micro,
            "seen_tokens": seen_tokens, "seen_examples": seen_examples,
            "next_epoch": epoch + int(micro_idx == micro_per_epoch),
            "skip_micro": 0 if micro_idx == micro_per_epoch else micro_idx,
            "seconds": elapsed_train(), "eval_seconds": eval_seconds,
            "peak": peak, "resumes": resumes,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)

    for epoch in range(first_epoch, tcfg["epochs"]):
        if step >= total_steps:
            break
        for micro_idx, batch in enumerate(batches(
            examples, tcfg["batch_size"], pad_id, shuffle=True, seed=tcfg["seed"] + epoch
        ), 1):
            if epoch == first_epoch and micro_idx <= skip_micro:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
            if n == 0:
                raise ValueError("train-батч не содержит токенов ответа")
            loss = model(**batch).loss
            if not math.isfinite(loss.item()):
                diverged = True
                raise RuntimeError(f"шаг {step}: loss={loss.item()} — обучение разошлось")
            # Сумма по токенам, затем нормировка градиента всей группы.
            # Это учитывает разные длины ответов и неполную последнюю группу.
            (loss * n).backward()
            accum_loss += loss.item() * n
            accum_tokens += n
            micro += 1
            seen_tokens += int(batch["attention_mask"].sum())
            seen_examples += batch["input_ids"].shape[0]
            peak = max(peak, allocated_bytes(device))
            if micro_idx % tcfg["grad_accum"] and micro_idx != micro_per_epoch:
                continue
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(accum_tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss / accum_tokens, 4)])
            accum_loss, accum_tokens = 0.0, 0
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                eval_started = time.perf_counter()
                val_loss = evaluate(model, val_blob["examples"], pad_id, device, eval_bs)
                eval_seconds += time.perf_counter() - eval_started
                curve_val.append([step, round(val_loss, 4)])
                peak = max(peak, allocated_bytes(device))
                print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.4f}, "
                      f"val {val_loss:.4f}", flush=True)
            else:
                print(f"  шаг {step}/{total_steps}: train {curve_train[-1][1]:.4f}", flush=True)
            stop = args.stop_after is not None and step == args.stop_after and step < total_steps
            if step % tcfg.get("checkpoint_every", tcfg["eval_every"]) == 0 or step == total_steps or stop:
                checkpoint(epoch, micro_idx)
            if stop:
                print(f"Контрольная точка шага {step} сохранена: {checkpoint_path}", flush=True)
                return
            if step >= total_steps:
                break
        if diverged or step >= total_steps:
            break
    seconds = elapsed_train()   # сумма активных отрезков, без оценки val и перерывов
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "variant": args.variant,
        "freeze_first": variant["freeze_first"],
        "model": params["model"]["name"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "base_val_loss": round(base_val, 4),
        "base_eval_seconds": round(base_eval_seconds, 1),
        "train_examples": len(examples),
        "val_examples": len(val_blob["examples"]),
        "seen_examples": seen_examples,
        "seen_tokens": seen_tokens,
        "micro_steps": micro,
        "final_val_loss": curve_val[-1][1] if curve_val else None,
        "diverged": diverged,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "seconds": round(seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "seconds_per_step": round(seconds / max(step, 1), 3),
        "train_tokens_per_sec": round(seen_tokens / seconds, 1) if seconds else 0,
        "peak_memory_mb": round(peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": fingerprint,
        "resumes": resumes,
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {step} шагов за {seconds:.0f} с; "
          f"пик памяти {metrics['peak_memory_mb']:.0f} МБ; адаптер {metrics['adapter_size_mb']} МБ -> {adapter_dir}")


if __name__ == "__main__":
    main()
