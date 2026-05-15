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


def load_mnist() -> np.ndarray:
    print("loading MNIST...")
    ds = cast(Any, fetch_openml("mnist_784", version=1, as_frame=False, parser="auto"))
    X = np.asarray(ds.data).astype(np.float32)
    print(f"  X.shape = {X.shape}  X.nbytes = {X.nbytes / 1e6:.1f} MB")
    return X


def warmup_jit(seed: int) -> None:
    print("warming up pynndescent JIT...")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((2_000, 32)).astype(np.float32)
    _ = get_umap_graph(X, random_state=seed, quiet=True)
    del X
    gc.collect()


def time_fit(
    X: np.ndarray, device: str, num_workers: int, epochs: int, seed: int
) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    _ = get_umap_graph(X, random_state=seed, quiet=True)
    graph_time = time.perf_counter() - t0

    reducer = GlassBoxUMAP(
        epochs=epochs,
        random_state=seed,
        quiet=True,
        num_workers=num_workers,
    )
    reducer.to(device)

    t0 = time.perf_counter()
    reducer.fit(X)
    fit_time = time.perf_counter() - t0
    train_time = max(fit_time - graph_time, 0.0)

    del reducer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return fit_time, graph_time, train_time


@app.command()
def main(
    output_dir: Path = Path(__file__).parent,
    seed: int = 22,
    epochs: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "workers.csv"
    meta_path = output_dir / "workers_meta.txt"

    X = load_mnist()
    warmup_jit(seed)

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")

    workers_grid = [0, 1, 2, 4, 8, 12]

    meta_path.write_text(
        "\n".join(
            [
                f"machine: {platform.platform()}",
                f"processor: {platform.processor()}",
                f"python: {platform.python_version()}",
                f"torch: {torch.__version__}",
                f"epochs: {epochs}",
                f"seed: {seed}",
                f"workers_grid: {workers_grid}",
                f"devices: {devices}",
                f"n_samples: {X.shape[0]}",
                f"n_features: {X.shape[1]}",
            ]
        )
        + "\n"
    )

    if not csv_path.exists():
        with csv_path.open("w") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    for device in devices:
        for w in workers_grid:
            print(f"\n=== device={device}  num_workers={w}  epochs={epochs} ===")
            fit_t, graph_t, train_t = time_fit(X, device, w, epochs, seed)
            row = {
                "device": device,
                "num_workers": w,
                "epochs": epochs,
                "fit_time_s": round(fit_t, 3),
                "graph_time_s": round(graph_t, 3),
                "train_time_s": round(train_t, 3),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            }
            print(f"  result: {row}")
            with csv_path.open("a") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


if __name__ == "__main__":
    app()
