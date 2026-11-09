"""Time Glass Box UMAP fits and check the maps they draw.

Each fit follows a pipeline that draws many maps of small and medium inputs: standardized
float32 features, ``random_state=0``, ``n_neighbors=min(15, max(2, n - 1))``, then
``fit_transform`` and ``compute_contributions``, on CPU.

Modes:

    warm        many fits in one process, after one warm-up fit
    cold        each fit in a new process, timed from before ``import glass_box_umap``
    throughput  several single-thread processes sharing the fits: machine-seconds per fit
    compare     the gap between two ``warm --save-dir`` runs, with quality and time side by side

Examples:

    python benchmarks/bench_fit.py warm --threads 8 --out new.jsonl --save-dir maps/new
    python benchmarks/bench_fit.py warm --sizes 5 50 200 --dims 11 --inputs data/*.npy
    python benchmarks/bench_fit.py warm --option neighbor_search=auto --out exact.jsonl
    python benchmarks/bench_fit.py cold --sizes 5 200 --threads 8
    python benchmarks/bench_fit.py throughput --procs 8 --sizes 5 10 20 50 100 200 500
    python benchmarks/bench_fit.py compare maps/v0.1.0 maps/new

``warm --save-dir`` writes each fit's coordinates and contributions, so runs of two versions of the
package (each in its own environment) can be compared with ``compare``. A gap of exactly 0 means the
same map, bit for bit.

Each fit reports:

- ``additivity``: the largest gap between the summed contributions and the coordinates;
- ``recall``: the share of each point's k nearest neighbors in the features (k = min(15, n - 1))
  that are also among its k nearest on the map.

The script runs against any version of the package, including 0.1.0. Timings only predict
production when they are taken on the machines the fits run on, with the CPU forced.
"""

from __future__ import annotations
import argparse
import ast
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SIZES = [5, 50, 200, 1_000, 5_000, 20_000, 50_000]
DEFAULT_DIMS = [9, 11, 24]


# -- inputs ---------------------------------------------------------------------------------------


def synthetic(n: int, d: int, seed: int = 0):
    """A mixture of anisotropic Gaussian clusters, as float32."""
    import numpy as np

    rng = np.random.default_rng([seed, n, d])
    n_clusters = int(min(8, max(1, n // 5)))
    centers = rng.normal(scale=4.0, size=(n_clusters, d))
    scales = rng.uniform(0.3, 1.5, size=(n_clusters, d))
    labels = rng.integers(n_clusters, size=n)
    X = centers[labels] + rng.normal(size=(n, d)) * scales[labels]
    return X.astype(np.float32)


def cases(args: argparse.Namespace) -> list[tuple[str, Any]]:
    """(name, loader) pairs. Loaders are called lazily so workers only load what they fit."""
    import numpy as np

    out: list[tuple[str, Any]] = []
    for d in args.dims:
        for n in args.sizes:
            out.append((f"synthetic-n{n:05d}-d{d:02d}", (lambda n=n, d=d: synthetic(n, d))))
    for path in args.inputs:
        p = Path(path)
        out.append((p.stem, (lambda p=p: np.load(p))))
    return out


def standardize(X):
    """Standardize each feature, leaving constant features centered."""
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1
    return (X - mean) / std


# -- one fit --------------------------------------------------------------------------------------


def parse_options(pairs: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            options[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            options[key] = raw
    return options


def fit_one(Z, options: dict[str, Any]) -> dict[str, Any]:
    """Fit Z and compute its contributions. Returns timings, quality and the arrays."""
    import numpy as np
    from glass_box_umap import GlassBoxUMAP

    k = min(15, max(2, len(Z) - 1))
    model = GlassBoxUMAP(random_state=0, quiet=True, n_neighbors=k, **options)
    model.to("cpu")  # The default device is CUDA or MPS when present; compare on CPU.
    t0 = time.perf_counter()
    Y = model.fit_transform(Z)
    t1 = time.perf_counter()
    C = model.compute_contributions(Z)
    t2 = time.perf_counter()
    return {
        "fit_s": round(t1 - t0, 4),
        "contrib_s": round(t2 - t1, 4),
        "map_s": round(t2 - t0, 4),
        "additivity": float(np.abs(C.sum(-1) - Y).max()),
        "Y": Y,
        "C": C,
    }


def recall(Z, Y, k: int = 15) -> float:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(Z) - 1)
    if k < 1:
        return float("nan")
    iz = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    iy = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(iz, iy, strict=True)]))


def warm_up(options: dict[str, Any]) -> float:
    """One small fit, so later timings exclude imports, numba compilation and first-call setup."""
    t0 = time.perf_counter()
    fit_one(standardize(synthetic(20, 11, seed=1)), options)
    return round(time.perf_counter() - t0, 3)


# -- reporting ------------------------------------------------------------------------------------


def environment(threads: int | None) -> dict[str, Any]:
    import importlib.metadata as md

    versions = {}
    for pkg in ("glass-box-umap", "torch", "numpy", "umap-learn", "pynndescent", "numba"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "mode": "environment",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "threads": threads,
        "instance_type": os.environ.get("INSTANCE_TYPE", ""),
        **versions,
    }


class Recorder:
    def __init__(self, out: str | None, tag: str | None) -> None:
        self.out = Path(out) if out else None
        self.tag = tag
        if self.out:
            self.out.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, rec: dict[str, Any]) -> None:
        if self.tag:
            rec = {"tag": self.tag, **rec}
        line = json.dumps(rec)
        print(line, flush=True)
        if self.out:
            with self.out.open("a") as f:
                f.write(line + "\n")


def set_threads(threads: int | None) -> None:
    import torch

    if threads:
        torch.set_num_threads(threads)


# -- modes ----------------------------------------------------------------------------------------


def run_warm(args: argparse.Namespace) -> None:
    import numpy as np

    set_threads(args.threads)
    emit = Recorder(args.out, args.tag)
    options = parse_options(args.option)
    emit({**environment(args.threads), "options": options})
    emit({"mode": "warmup", "s": warm_up(options)})
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for name, load in cases(args):
        Z = standardize(load())
        for rep in range(args.repeats):
            res = fit_one(Z, options)
            Y, C = res.pop("Y"), res.pop("C")
            if save_dir and rep == 0:
                np.savez(save_dir / f"{name}.npz", Y=Y, C=C)
            emit(
                {
                    "mode": "warm",
                    "case": name,
                    "n": int(Z.shape[0]),
                    "d": int(Z.shape[1]),
                    "rep": rep,
                    **res,
                    "recall": round(recall(Z, Y), 4),
                }
            )


def run_cold_child(args: argparse.Namespace) -> None:
    """Runs in a new process. Imports numpy and torch first, then times the rest."""
    t_start = time.perf_counter()
    import numpy as np  # noqa: F401
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    loaders = dict(cases(args))
    Z = standardize(loaders[args.case]())
    options = parse_options(args.option)
    t0 = time.perf_counter()
    import glass_box_umap  # noqa: F401

    t1 = time.perf_counter()
    res = fit_one(Z, options)
    res.pop("Y")
    res.pop("C")
    rec = {
        "pre_s": round(t0 - t_start, 3),
        "import_s": round(t1 - t0, 3),
        "first_map_s": round(time.perf_counter() - t0, 3),
        **res,
    }
    print("RESULT " + json.dumps(rec), flush=True)


def child_argv(args: argparse.Namespace, case: str) -> list[str]:
    argv = [sys.executable, __file__, "_cold_child", "--case", case]
    argv += ["--sizes", *map(str, args.sizes), "--dims", *map(str, args.dims)]
    if args.inputs:
        argv += ["--inputs", *args.inputs]
    if args.threads:
        argv += ["--threads", str(args.threads)]
    for opt in args.option:
        argv += ["--option", opt]
    return argv


def run_cold(args: argparse.Namespace) -> None:
    emit = Recorder(args.out, args.tag)
    emit({**environment(args.threads), "options": parse_options(args.option)})
    env = dict(os.environ)
    if args.threads:
        env.update({"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)})
    for name, load in cases(args):
        n = int(load().shape[0])
        for rep in range(args.repeats):
            t0 = time.perf_counter()
            p = subprocess.run(child_argv(args, name), capture_output=True, text=True, env=env)
            wall = time.perf_counter() - t0
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            rec = json.loads(lines[-1][len("RESULT ") :]) if lines else {}
            emit(
                {
                    "mode": "cold",
                    "case": name,
                    "n": n,
                    "rep": rep,
                    "process_s": round(wall, 3),
                    "rc": p.returncode,
                    **rec,
                    **({"stderr": p.stderr[-2000:]} if p.returncode else {}),
                }
            )


def _throughput_worker(args: argparse.Namespace, queue, done, ready, start) -> None:
    import torch

    torch.set_num_threads(1)
    options = parse_options(args.option)
    loaders = dict(cases(args))
    warm_up(options)
    ready.put(1)
    start.wait()
    while (name := queue.get()) is not None:
        Z = standardize(loaders[name]())
        t0 = time.perf_counter()
        res = fit_one(Z, options)
        done.put(
            {
                "case": name,
                "n": int(Z.shape[0]),
                "s": round(time.perf_counter() - t0, 4),
                "additivity": res["additivity"],
            }
        )
    done.put(None)


def run_throughput(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    emit = Recorder(args.out, args.tag)
    emit({**environment(1), "options": parse_options(args.option), "procs": args.procs})
    names = [name for name, _ in cases(args)] * args.repeats
    sizes = {name: int(load().shape[0]) for name, load in cases(args)}
    names.sort(key=lambda name: -sizes[name])  # Largest first, so the tail is short.
    ctx = mp.get_context("spawn")
    queue, done, ready, start = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Event()
    procs = [
        ctx.Process(target=_throughput_worker, args=(args, queue, done, ready, start))
        for _ in range(args.procs)
    ]
    for p in procs:
        p.start()
    for _ in procs:
        ready.get()
    for name in names:
        queue.put(name)
    for _ in procs:
        queue.put(None)
    t0 = time.perf_counter()
    start.set()
    recs, finished = [], 0
    while finished < args.procs:
        rec = done.get()
        if rec is None:
            finished += 1
        else:
            recs.append(rec)
    wall = time.perf_counter() - t0
    for p in procs:
        p.join()
    emit(
        {
            "mode": "throughput",
            "procs": args.procs,
            "fits": len(recs),
            "wall_s": round(wall, 3),
            "machine_s_per_fit": round(wall / max(1, len(recs)), 4),
            "mean_fit_s": round(sum(r["s"] for r in recs) / max(1, len(recs)), 4),
            "max_additivity": max((r["additivity"] for r in recs), default=None),
        }
    )


def run_compare(args: argparse.Namespace) -> None:
    import numpy as np

    ref_dir, new_dir = Path(args.reference), Path(args.candidate)
    names = sorted(p.stem for p in new_dir.glob("*.npz") if (ref_dir / p.name).exists())
    if not names:
        raise SystemExit(f"No maps in both {ref_dir} and {new_dir}.")
    print("| case | rows | coordinate gap | contribution gap |")
    print("|---|---|---|---|")
    worst = 0.0
    for name in names:
        ref, new = np.load(ref_dir / f"{name}.npz"), np.load(new_dir / f"{name}.npz")
        gap_y = float(np.abs(ref["Y"] - new["Y"]).max())
        gap_c = float(np.abs(ref["C"] - new["C"]).max())
        worst = max(worst, gap_y, gap_c)
        print(f"| {name} | {len(ref['Y']):,} | {gap_y:.3g} | {gap_c:.3g} |")
    print(f"\nLargest gap: {worst:.3g} over {len(names)} maps.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
        p.add_argument("--dims", type=int, nargs="*", default=DEFAULT_DIMS)
        p.add_argument("--inputs", nargs="*", default=[], help="Real feature matrices (.npy)")
        p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
        p.add_argument(
            "--option",
            action="append",
            default=[],
            help="Extra GlassBoxUMAP keyword, as key=value (repeatable)",
        )
        p.add_argument("--out", help="Append JSON lines to this file")
        p.add_argument("--tag", help="Label added to every record")
        p.add_argument("--repeats", type=int, default=1)

    common(warm := sub.add_parser("warm", help="Many fits in one process"))
    warm.add_argument("--save-dir", help="Save each map's coordinates and contributions here")
    common(sub.add_parser("cold", help="Each fit in a new process"))
    common(tp := sub.add_parser("throughput", help="Single-thread processes sharing the fits"))
    tp.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    common(child := sub.add_parser("_cold_child"))
    child.add_argument("--case", required=True)
    cmp = sub.add_parser("compare", help="Gap between two warm --save-dir runs")
    cmp.add_argument("reference")
    cmp.add_argument("candidate")

    args = ap.parse_args()
    {
        "warm": run_warm,
        "cold": run_cold,
        "throughput": run_throughput,
        "_cold_child": run_cold_child,
        "compare": run_compare,
    }[args.mode](args)


if __name__ == "__main__":
    main()
