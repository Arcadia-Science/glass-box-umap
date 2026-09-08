# Choosing a training batch size

The fastest batch is not necessarily the largest one that fits in accelerator
memory. Very large batches reduce the number of optimizer updates in each epoch
and can change the embedding even when they improve raw throughput.

Use `recommend_batch_size` with the same model configuration and data that you
intend to train:

```python
from glass_box_umap import GlassBoxUMAP, recommend_batch_size

model = GlassBoxUMAP(precision="bf16-mixed", random_state=42)
result = recommend_batch_size(X, model=model)
print(result.summary())

model.batch_size = result.recommended_batch_size
model.fit(X)
```

The default `mode="quick"` performs a short throughput scan and then checks two
matched training epochs for the most promising batch sizes. A candidate is
rejected if it provides fewer than 20 optimizer steps per epoch or its second
epoch loss is more than 5% worse than the conservative reference. Among the
remaining candidates, the smallest batch within 5% of peak throughput is
recommended.

For a costly production run, use `mode="thorough"`. It tests a denser candidate
grid for five epochs and compares both the mean and final loss against the
reference. You can also supply explicit candidates:

```python
result = recommend_batch_size(
    X,
    model=model,
    mode="thorough",
    batch_sizes=[8_192, 16_384, 24_576, 32_768],
)
```

Peak accelerator memory is reported as a diagnostic, but memory utilization is
not a selection target. Rerun the benchmark after materially changing the GPU,
encoder, feature dimensions, precision, or negative-sampling rate.
