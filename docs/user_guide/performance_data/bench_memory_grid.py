import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
import typer

app = typer.Typer(pretty_exceptions_enable=False)

CELL_SCRIPT = Path(__file__).parent / "bench_memory_cell.py"

CELL_FIELDS = [
    "n_samples",
    "n_features",
    "ok",
    "exit_code",
    "elapsed_s",
    "peak_rss_bytes",
    "external_peak_rss_bytes",
    "num_edges",
    "input_bytes",
]


def run_cell(
    n_samples: int,
    n_features: int,
    seed: int,
    timeout_s: float,
    sample_interval_s: float,
) -> dict[str, int | float | bool | None]:
    out_path = Path(f"/tmp/bench_memory_cell_{n_samples}_{n_features}.json")
    if out_path.exists():
        out_path.unlink()

    env = {**os.environ, "BENCH_OUT": str(out_path)}
    cmd = [sys.executable, str(CELL_SCRIPT), str(n_samples), str(n_features), str(seed)]

    proc = subprocess.Popen(cmd, env=env)
    parent = psutil.Process(proc.pid)
    external_peak = 0
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout_s:
            proc.kill()
            proc.wait()
            break
        try:
            members = [parent] + parent.children(recursive=True)
            cur = sum(c.memory_info().rss for c in members if c.is_running())
            if cur > external_peak:
                external_peak = cur
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        time.sleep(sample_interval_s)
    rc = proc.returncode

    row: dict[str, int | float | bool | None] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "ok": False,
        "exit_code": rc,
        "elapsed_s": None,
        "peak_rss_bytes": None,
        "external_peak_rss_bytes": external_peak,
        "num_edges": None,
        "input_bytes": None,
    }

    if rc == 0 and out_path.exists():
        result = json.loads(out_path.read_text())
        row.update(
            ok=True,
            elapsed_s=result["elapsed_s"],
            peak_rss_bytes=result["peak_rss_bytes"],
            num_edges=result["num_edges"],
            input_bytes=result["input_bytes"],
        )
    return row


@app.command()
def main(
    output_dir: Path = Path(__file__).parent,
    seed: int = 22,
    timeout_s: float = 900.0,
    sample_interval_s: float = 0.05,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "memory_grid.csv"
    meta_path = output_dir / "memory_grid_meta.txt"

    sample_grid = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
    feature_grid = [16, 64, 256, 1024]

    meta_path.write_text(
        "\n".join(
            [
                f"machine: {platform.platform()}",
                f"processor: {platform.processor()}",
                f"python: {platform.python_version()}",
                f"torch: {torch.__version__}",
                f"total_ram_gb: {psutil.virtual_memory().total / 1e9:.1f}",
                f"timeout_s: {timeout_s}",
                f"seed: {seed}",
                f"mode: graph_only",
                f"sample_grid: {sample_grid}",
                f"feature_grid: {feature_grid}",
            ]
        )
        + "\n"
    )

    if not csv_path.exists():
        with csv_path.open("w") as f:
            csv.DictWriter(f, fieldnames=CELL_FIELDS).writeheader()

    for n in sample_grid:
        for d in feature_grid:
            print(f"\n=== N={n:,}  D={d} ===  (timeout={timeout_s:.0f}s)")
            t0 = time.perf_counter()
            row = run_cell(n, d, seed, timeout_s, sample_interval_s)
            wall = time.perf_counter() - t0
            ext_gb = row["external_peak_rss_bytes"] / 1e9 if row["external_peak_rss_bytes"] else 0.0
            tag = "ok" if row["ok"] else f"FAIL exit={row['exit_code']}"
            print(f"  {tag}  external_peak_rss={ext_gb:.2f} GB  wall={wall:.1f}s")
            with csv_path.open("a") as f:
                csv.DictWriter(f, fieldnames=CELL_FIELDS).writerow(row)


if __name__ == "__main__":
    app()
