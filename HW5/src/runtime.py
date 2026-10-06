"""Устройство, тип весов, сид и память — всё, что зависит от железа.

Код обучения не знает, на чём он запущен: устройство выбирается здесь по
params.yaml. Смена железа — правка конфига, а не кода.
"""

import os
import random

import numpy as np
import psutil
import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(name: str) -> torch.device:
    """auto -> cuda, если есть; иначе mps; иначе cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise SystemExit(f"model.dtype = {name!r}, допустимо: {sorted(DTYPES)}")
    return DTYPES[name]


def set_seed(seed: int) -> None:
    """Один сид на всё, что тянет случайность: инициализацию A в LoRA,
    dropout, порядок примеров. Без него два прогона с одним конфигом —
    два разных эксперимента, и сравнивать их нельзя."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def allocated_bytes(device: torch.device) -> int:
    """Память, занятая тензорами сейчас. На ускорителе — его аллокатор, а не RSS:
    урок ДЗ 2, на Apple Silicon RSS не видит буферы Metal."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated()
    return psutil.Process().memory_info().rss


def memory_metric(device: torch.device) -> str:
    return {
        "mps": "torch.mps.driver_allocated_memory",
        "cuda": "torch.cuda.memory_allocated",
    }.get(device.type, "RSS процесса")


def rng_state(device: torch.device) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
