"""Run the workers sweep on a Modal H100 with 32 CPU cores.

Usage:
    modal run docs/user_guide/performance_data/bench_workers_h100_modal.py

Writes the resulting CSV to ``docs/user_guide/performance_data/workers_h100.csv``
relative to the project root. The format matches ``workers.csv`` so that the
scaling notebook can plot all three lines (MPS, CPU, H100) together.
"""

import csv
import gc
import io
import time
from pathlib import Path

import modal

app = modal.App("glass-box-umap-bench-h100")

DEPS = [
    "numpy>=1.26.4,<2",
    "pandas>=2.2.3,<3",
    "leidenalg>=0.10.2,<0.11",
    "lightning-utilities>=0.15.2,<0.16",
    "dill>=0.4.0,<0.5",
    "einops>=0.8.1,<0.9",
    "torch>=2.5.1",
    "pytorch-lightning>=2.6.0",
    "umap-learn>=0.5.7",
    "pynndescent>=0.5.13",
    "scikit-learn>=1.6.0",
    "tensorboard>=2.20.0",
    "packaging>=23.0",
    "psutil>=5.9",
    "typer>=0.9",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*DEPS)
    .add_local_python_source("glass_box_umap")
)

WORKERS_GRID = [0, 1, 2, 4, 8, 12]
FIELDS = [
    "device",
    "num_workers",
    "epochs",
    "fit_time_s",
    "graph_time_s",
    "train_time_s",
    "n_samples",
    "n_features",
]


@app.function(
    image=image,
    gpu="H100",
    cpu=(32.0, 32.0),
    memory=32 * 1024,
    timeout=3600,
)
def run_workers_bench(epochs: int = 1, seed: int = 22) -> str:
    from typing import Any, cast

    import numpy as np
    import torch
    from sklearn.datasets import fetch_openml

    from glass_box_umap import GlassBoxUMAP
    from glass_box_umap.parametric_umap.graph import get_umap_graph

    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device name: {torch.cuda.get_device_name(0)}")
        print(f"device count: {torch.cuda.device_count()}")

    print("loading MNIST...")
    ds = cast(Any, fetch_openml("mnist_784", version=1, as_frame=False, parser="auto"))
    X = np.asarray(ds.data).astype(np.float32)
    print(f"X.shape={X.shape} X.nbytes={X.nbytes / 1e6:.1f} MB")

    print("warming up pynndescent JIT...", flush=True)
    rng = np.random.default_rng(seed)
    Xw = rng.standard_normal((2_000, 32)).astype(np.float32)
    _ = get_umap_graph(Xw, random_state=seed, quiet=True)
    del Xw
    gc.collect()

    print("warming up CUDA (1-epoch fit on small data)...", flush=True)
    Xw2 = rng.standard_normal((5_000, 64)).astype(np.float32)
    warmup_reducer = GlassBoxUMAP(epochs=1, random_state=seed, quiet=False, num_workers=0)
    warmup_reducer.to("cuda")
    warmup_reducer.fit(Xw2)
    del warmup_reducer, Xw2
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA warmup done.", flush=True)

    rows: list[dict[str, int | float | str]] = []
    for w in WORKERS_GRID:
        device = "cuda"
        print(f"\n=== device={device}  num_workers={w}  epochs={epochs} ===", flush=True)

        t0 = time.perf_counter()
        _ = get_umap_graph(X, random_state=seed, quiet=True)
        graph_time = time.perf_counter() - t0
        print(f"  graph built in {graph_time:.2f}s; starting fit...", flush=True)

        reducer = GlassBoxUMAP(
            epochs=epochs,
            random_state=seed,
            quiet=False,
            num_workers=w,
        )
        reducer.to(device)

        t0 = time.perf_counter()
        reducer.fit(X)
        fit_time = time.perf_counter() - t0
        train_time = max(fit_time - graph_time, 0.0)

        del reducer
        gc.collect()
        torch.cuda.empty_cache()

        row: dict[str, int | float | str] = {
            "device": device,
            "num_workers": w,
            "epochs": epochs,
            "fit_time_s": round(fit_time, 3),
            "graph_time_s": round(graph_time, 3),
            "train_time_s": round(train_time, 3),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }
        print(f"  result: {row}", flush=True)
        rows.append(row)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str = "docs/user_guide/performance_data/workers_h100.csv") -> None:
    csv_data = run_workers_bench.remote()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_data)
    print(f"wrote {out_path}")
