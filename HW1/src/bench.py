"""Замер load time, tokens/sec и peak RSS — раздельно."""

import json
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

from src.config import load_params
from src.model import generate, load_model, set_seed


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    params = load_params()
    set_seed(params["generate"]["seed"])
    prompt = params["bench"]["prompt"]

    peak = [0.0]
    stop = threading.Event()

    def track():
        while not stop.is_set():
            peak[0] = max(peak[0], peak_rss_mb())
            time.sleep(0.05)

    tracker = threading.Thread(target=track, daemon=True)
    tracker.start()

    t0 = time.perf_counter()
    tokenizer, model = load_model(params)
    load_time = time.perf_counter() - t0
    peak[0] = max(peak[0], peak_rss_mb())

    for _ in range(params["bench"]["warmup_runs"]):
        generate(tokenizer, model, params, prompt)

    speeds = []
    for _ in range(params["bench"]["measure_runs"]):
        t_start = time.perf_counter()
        _, n_tokens = generate(tokenizer, model, params, prompt)
        speeds.append(n_tokens / (time.perf_counter() - t_start))
        peak[0] = max(peak[0], peak_rss_mb())

    stop.set()
    tracker.join(timeout=1.0)

    report = {
        "model": params["model"]["name"],
        "device": str(model.device),
        "dtype": params["model"]["dtype"],
        "load_time_sec": round(load_time, 2),
        "tokens_per_sec": round(statistics.median(speeds), 2),
        "tokens_per_sec_all": [round(s, 2) for s in speeds],
        "peak_rss_mb": round(peak[0], 1),
    }

    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
