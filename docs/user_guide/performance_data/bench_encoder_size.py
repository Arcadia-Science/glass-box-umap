"""Local CPU + MPS sweep: 133K vs 5M encoder, 1 epoch, full MNIST.

Pairs with ``bench_encoder_size_h100_modal.py`` (the H100 row of the same
experiment). The point is to isolate the role of GPU compute vs. host loop
overhead by scaling encoder parameters ~37x and watching wall clock.

Outputs CSV to ``performance_data/encoder_size.csv`` (next to this script).
"""

import csv
import gc
import platform
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import typer
from glass_box_umap import GlassBoxUMAP
from glass_box_umap.parametric_umap.graph import get_umap_graph
from sklearn.datasets import fetch_openml

app = typer.Typer(pretty_exceptions_enable=False)

NUM_WORKERS = 2
BATCH_SIZE = 10_000

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


def load_mnist() -> np.ndarray:
    print("loading MNIST...")
    ds = cast(Any, fetch_openml("mnist_784", version=1, as_frame=False, parser="auto"))
    X = np.asarray(ds.data).astype(np.float32)
    print(f"  X.shape = {X.shape}  X.nbytes = {X.nbytes / 1e6:.1f} MB")
    return X


def warmup_jit(seed: int) -> None:
    print("warming up pynndescent JIT...")
    rng = np.random.default_rng(seed)
    Xw = rng.standard_normal((2_000, 32)).astype(np.float32)
    _ = get_umap_graph(Xw, random_state=seed, quiet=True)
    del Xw
    gc.collect()


def run_fit(
    X: np.ndarray,
    device: str,
    encoder_label: str,
    encoder_kwargs: dict[str, int],
    seed: int,
    epochs: int,
) -> dict[str, int | float | str]:
    print(f"\n=== device={device}  encoder={encoder_label}  epochs={epochs} ===")

    t0 = time.perf_counter()
    graph = get_umap_graph(X, random_state=seed, quiet=True)
    graph_time = time.perf_counter() - t0
    num_edges = int(graph.nnz)
    del graph
    gc.collect()

    reducer = GlassBoxUMAP(
        epochs=epochs,
        batch_size=BATCH_SIZE,
        random_state=seed,
        quiet=False,
        num_workers=NUM_WORKERS,
        encoder_kwargs=encoder_kwargs,
    )
    reducer.to(device)

    t0 = time.perf_counter()
    reducer.fit(X)
    fit_time = time.perf_counter() - t0
    train_time = max(fit_time - graph_time, 0.0)

    n_params = sum(p.numel() for p in reducer._fitted_model.parameters())

    del reducer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    row: dict[str, int | float | str] = {
        "device": device,
        "encoder": encoder_label,
        "n_params": int(n_params),
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "fit_time_s": round(fit_time, 3),
        "graph_time_s": round(graph_time, 3),
        "train_time_s": round(train_time, 3),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "num_edges": num_edges,
    }
    print(f"  result: {row}")
    return row


@app.command()
def main(
    output_dir: Path = Path(__file__).parent,
    seed: int = 22,
    epochs: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "encoder_size.csv"
    meta_path = output_dir / "encoder_size_meta.txt"

    devices: list[str] = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")

    meta_path.write_text(
        "\n".join(
            [
                f"machine: {platform.platform()}",
                f"processor: {platform.processor()}",
                f"python: {platform.python_version()}",
                f"torch: {torch.__version__}",
                f"epochs: {epochs}",
                f"seed: {seed}",
                f"batch_size: {BATCH_SIZE}",
                f"num_workers: {NUM_WORKERS}",
                f"devices: {devices}",
                f"encoder_variants: {sorted(ENCODER_VARIANTS)}",
            ]
        )
        + "\n"
    )

    if not csv_path.exists():
        with csv_path.open("w") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    X = load_mnist()
    warmup_jit(seed)

    for encoder_label, encoder_kwargs in ENCODER_VARIANTS.items():
        for device in devices:
            row = run_fit(X, device, encoder_label, encoder_kwargs, seed, epochs)
            with csv_path.open("a") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


if __name__ == "__main__":
    app()
