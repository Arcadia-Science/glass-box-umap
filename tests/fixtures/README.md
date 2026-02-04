# Test Fixtures

This directory contains pre-generated data for unit tests.

## Files

- `mnist_images.pt` - Subset of MNIST images as a float32 tensor of shape `(N, 784)`
- `mnist_labels.pt` - Corresponding labels as an int64 tensor of shape `(N,)`

## Regenerating Fixtures

To regenerate the MNIST fixtures:

```bash
python tests/fixtures/create_mnist.py --num-samples 100
```
