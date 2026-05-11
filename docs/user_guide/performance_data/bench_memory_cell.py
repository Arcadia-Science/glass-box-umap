import json
import os
import sys
import threading
import time

import numpy as np
import psutil
from glass_box_umap.parametric_umap.graph import get_umap_graph

n_samples = int(sys.argv[1])
n_features = int(sys.argv[2])
seed = int(sys.argv[3])

proc = psutil.Process()
peak_rss = [proc.memory_info().rss]
stop = threading.Event()


def sampler() -> None:
    while not stop.is_set():
        cur = proc.memory_info().rss
        if cur > peak_rss[0]:
            peak_rss[0] = cur
        stop.wait(0.05)


t = threading.Thread(target=sampler, daemon=True)
t.start()

n_centers = 20
rng = np.random.default_rng(seed)
centers = (rng.standard_normal((n_centers, n_features)) * 5.0).astype(np.float32)
labels = rng.integers(0, n_centers, size=n_samples)
noise = rng.standard_normal((n_samples, n_features)).astype(np.float32)
X = centers[labels] + noise

t0 = time.perf_counter()
graph = get_umap_graph(X, random_state=seed, quiet=True)
elapsed = time.perf_counter() - t0
num_edges = int(graph.nnz)

stop.set()
t.join()

out_path = os.environ["BENCH_OUT"]
result = {
    "n_samples": n_samples,
    "n_features": n_features,
    "input_bytes": int(X.nbytes),
    "elapsed_s": elapsed,
    "peak_rss_bytes": peak_rss[0],
    "num_edges": num_edges,
}
with open(out_path, "w") as f:
    json.dump(result, f)
