"""Modal T4 sweep: 133K vs 5M encoder, 1 epoch, full MNIST.

Sister script to ``bench_encoder_size_h100_modal.py``; same protocol but on
the older, smaller T4 (16 GB) so the encoder-size matrix spans a wider GPU
range.

Usage:
    modal run docs/user_guide/performance_data/bench_encoder_size_t4_modal.py

Writes CSV to ``docs/user_guide/performance_data/encoder_size_t4.csv``.
"""

import csv
import gc
import io
import time
from pathlib import Path

import modal

app = modal.App("glass-box-umap-bench-encoder-size-t4")

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

NUM_WORKERS = 8
BATCH_SIZE = 10_000
EPOCHS = 1

ENCODER_VARIANTS: dict[str, dict[str, int]] = {
    "133K": {"hidden_size": 128, "n_hidden_layers": 3},
    "463K": {"hidden_size": 256, "n_hidden_layers": 5},
    "2M": {"hidden_size": 512, "n_hidden_layers": 7},
    "5M": {"hidden_size": 1024, "n_hidden_layers": 5},
    "18M": {"hidden_size": 2048, "n_hidden_layers": 5},
}

FIELDS = [
    "device",
    "encoder",
    "n_params",
    "epochs",
    "batch_size",
    "num_workers",
    "fit_time_s",
    "graph_time_s",
    "train_time_s",
    "n_samples",
    "n_features",
    "num_edges",
]


@app.function(
    image=image,
    gpu="T4",
    cpu=(8.0, 8.0),
    memory=16 * 1024,
    timeout=3600,
)
def run_bench(seed: int = 22) -> str:
    from typing import Any, cast

    import numpy as np
    import torch
    from sklearn.datasets import fetch_openml

    from glass_box_umap import GlassBoxUMAP
    from glass_box_umap.parametric_umap.graph import get_umap_graph

    print(f"torch {torch.__version__}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device name: {torch.cuda.get_device_name(0)}", flush=True)

    print("loading MNIST...", flush=True)
    ds = cast(Any, fetch_openml("mnist_784", version=1, as_frame=False, parser="auto"))
    X = np.asarray(ds.data).astype(np.float32)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    print(f"X.shape={X.shape}", flush=True)

    print("warming up pynndescent JIT...", flush=True)
    rng = np.random.default_rng(seed)
    Xw = rng.standard_normal((2_000, 32)).astype(np.float32)
    _ = get_umap_graph(Xw, random_state=seed, quiet=True)
    del Xw
    gc.collect()

    print("warming up CUDA (1-epoch fit on small data)...", flush=True)
    Xw2 = rng.standard_normal((5_000, 64)).astype(np.float32)
    warmup = GlassBoxUMAP(epochs=1, random_state=seed, quiet=True, num_workers=0)
    warmup.to("cuda")
    warmup.fit(Xw2)
    del warmup, Xw2
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA warmup done.", flush=True)

    rows: list[dict[str, int | float | str]] = []
    for encoder_label, encoder_kwargs in ENCODER_VARIANTS.items():
        print(
            f"\n=== device=cuda  encoder={encoder_label}  epochs={EPOCHS} ===",
            flush=True,
        )

        t0 = time.perf_counter()
        graph = get_umap_graph(X, random_state=seed, quiet=True)
        graph_time = time.perf_counter() - t0
        num_edges = int(graph.nnz)
        print(
            f"  graph built in {graph_time:.2f}s ({num_edges:,} edges); starting fit...",
            flush=True,
        )
        del graph
        gc.collect()

        reducer = GlassBoxUMAP(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            random_state=seed,
            quiet=False,
            num_workers=NUM_WORKERS,
            encoder_kwargs=encoder_kwargs,
        )
        reducer.to("cuda")

        t0 = time.perf_counter()
        reducer.fit(X)
        fit_time = time.perf_counter() - t0
        train_time = max(fit_time - graph_time, 0.0)

        n_params = sum(p.numel() for p in reducer._fitted_model.parameters())

        del reducer
        gc.collect()
        torch.cuda.empty_cache()

        row: dict[str, int | float | str] = {
            "device": "cuda",
            "encoder": encoder_label,
            "n_params": int(n_params),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "fit_time_s": round(fit_time, 3),
            "graph_time_s": round(graph_time, 3),
            "train_time_s": round(train_time, 3),
            "n_samples": n_samples,
            "n_features": n_features,
            "num_edges": num_edges,
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
def main(out: str = "docs/user_guide/performance_data/encoder_size_t4.csv") -> None:
    csv_data = run_bench.remote()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_data)
    print(f"wrote {out_path}")
