# Performance benchmark data

This directory holds the benchmark scripts and the CSV outputs that the [performance notebook](../performance.ipynb) reads. The notebook itself does not run any benchmarks; it loads these CSVs and renders plots, so the data needs to live here for the page to render.

## Regenerating the data

All local scripts default to writing their CSVs back into this directory, so you can run them from anywhere. Modal scripts write the CSV to the same default path via the `out` argument; running them from the project root keeps that path consistent.

Note, we used Modal (`uv pip install modal`; requires account and authentication) to perform the GPU tests.

### Memory frontier — `memory_grid.csv`

Sweeps `(n_samples, n_features)` for the nearest-neighbor graph build and records peak resident set size (RSS) on each cell. Each cell runs in its own subprocess so that an OS-level kill (kernel reclaiming memory under pressure) takes down only that cell.

```bash
python docs/user_guide/performance_data/bench_memory_grid.py
```

### `num_workers` sweep — `workers.csv`, `workers_t4.csv`, `workers_h100.csv`

Fits Glass Box UMAP on full MNIST (70k rows, 784 features) for one epoch at each value of `num_workers`, recording fit time, graph time, and train time.

```bash
python docs/user_guide/performance_data/bench_workers.py
modal run docs/user_guide/performance_data/bench_workers_t4_modal.py
modal run docs/user_guide/performance_data/bench_workers_h100_modal.py
```

### Encoder-size sweep — `encoder_size.csv`, `encoder_size_t4.csv`, `encoder_size_h100.csv`

Fits Glass Box UMAP on full MNIST for one epoch at each of five encoder sizes (133K, 463K, 2M, 5M, 18M parameters), reached by widening `hidden_size` and deepening `n_hidden_layers`. Records fit time and parameter count per row. Uses device-near-optimal `num_workers` (2 on CPU and MPS, 8 on T4 and H100) so that the comparison is apples-to-apples across devices.

```bash
python docs/user_guide/performance_data/bench_encoder_size.py
modal run docs/user_guide/performance_data/bench_encoder_size_t4_modal.py
modal run docs/user_guide/performance_data/bench_encoder_size_h100_modal.py
```

## Running everything

The full sweep, in dependency order:

```bash
python docs/user_guide/performance_data/bench_memory_grid.py
python docs/user_guide/performance_data/bench_workers.py
python docs/user_guide/performance_data/bench_encoder_size.py
modal run docs/user_guide/performance_data/bench_workers_t4_modal.py
modal run docs/user_guide/performance_data/bench_workers_h100_modal.py
modal run docs/user_guide/performance_data/bench_encoder_size_t4_modal.py
modal run docs/user_guide/performance_data/bench_encoder_size_h100_modal.py
```

Once all seven CSVs are in place, re-execute the notebook to bake the new plots:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/user_guide/performance.ipynb
```
