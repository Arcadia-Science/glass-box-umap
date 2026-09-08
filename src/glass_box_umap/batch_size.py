from __future__ import annotations
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from .parametric_umap.core import ParametricUMAP, _to_numpy_float32
from .parametric_umap.data import UMAPDataset
from .parametric_umap.lightning.data import UMAPDataModule
from .parametric_umap.loss import umap_loss

BenchmarkMode = Literal["quick", "thorough"]


@dataclass
class BatchSizeCandidate:
    """Measurements and eligibility decision for one batch size."""

    batch_size: int
    steps_per_epoch: int
    status: str = "pending"
    edges_per_second: float | None = None
    peak_accelerator_memory_bytes: int | None = None
    epoch_losses: tuple[float, ...] = ()
    loss_ratio: float | None = None
    eligible: bool = False
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BatchSizeRecommendation:
    """Result returned by :func:`recommend_batch_size`."""

    recommended_batch_size: int
    mode: BenchmarkMode
    device: str
    precision: str
    seed: int
    graph_edges: int
    graph_build_seconds: float
    minimum_steps_per_epoch: int
    loss_tolerance: float
    throughput_tolerance: float
    candidates: tuple[BatchSizeCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the benchmark."""
        return asdict(self)

    def summary(self) -> str:
        """Return a compact human-readable recommendation."""
        chosen = next(
            candidate
            for candidate in self.candidates
            if candidate.batch_size == self.recommended_batch_size
        )
        throughput = (
            "unknown"
            if chosen.edges_per_second is None
            else f"{chosen.edges_per_second:,.0f} edges/s"
        )
        return (
            f"Recommended batch_size={self.recommended_batch_size:,} "
            f"({throughput}, {chosen.steps_per_epoch} steps/epoch, "
            f"precision={self.precision}, device={self.device})."
        )


@dataclass(frozen=True)
class _PrecisionConfig:
    parameter_dtype: torch.dtype
    autocast_dtype: torch.dtype | None = None
    use_grad_scaler: bool = False


def _precision_config(precision: str, device: torch.device) -> _PrecisionConfig:
    normalized = precision.lower()
    if normalized in {"32", "32-true"}:
        return _PrecisionConfig(torch.float32)
    if normalized in {"64", "64-true"}:
        return _PrecisionConfig(torch.float64)
    if normalized in {"bf16", "bf16-true"}:
        return _PrecisionConfig(torch.bfloat16)
    if normalized in {"16", "16-true"}:
        return _PrecisionConfig(torch.float16)
    if normalized == "bf16-mixed":
        return _PrecisionConfig(torch.float32, torch.bfloat16)
    if normalized == "16-mixed":
        if device.type != "cuda":
            raise ValueError("16-mixed batch-size benchmarking requires a CUDA device")
        return _PrecisionConfig(torch.float32, torch.float16, use_grad_scaler=True)
    raise ValueError(
        "Unsupported benchmark precision. Use 32-true, 64-true, 16-true, "
        "bf16-true, 16-mixed, or bf16-mixed."
    )


def _rounded_cap(value: int) -> int:
    if value < 1024:
        return max(value, 1)
    return max((value // 1024) * 1024, 1024)


def _adaptive_batch_sizes(
    graph_edges: int,
    configured_batch_size: int,
    minimum_steps_per_epoch: int,
    mode: BenchmarkMode,
) -> list[int]:
    effective_floor = min(minimum_steps_per_epoch, graph_edges)
    cap = _rounded_cap(max(graph_edges // max(effective_floor, 1), 1))
    anchor = min(max(configured_batch_size, 1), cap)

    if mode == "quick":
        values = [anchor // 2, anchor, anchor * 2, anchor * 4, cap]
    else:
        values = [
            anchor // 2,
            anchor,
            anchor * 3 // 2,
            anchor * 2,
            anchor * 5 // 2,
            anchor * 3,
            anchor * 4,
            cap,
        ]
    return sorted({max(1, min(value, cap)) for value in values})


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return None


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cannot allocate memory" in message


def _release_accelerator_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _prepare_data(
    X: NDArray[np.floating] | Tensor,
    model: ParametricUMAP,
) -> tuple[NDArray[np.float32], UMAPDataset, float]:
    prepared = _to_numpy_float32(X)
    prepared = prepared - prepared.mean(axis=0)
    if model.pca_components is not None:
        if prepared.ndim != 2:
            raise ValueError("PCA batch-size benchmarking requires two-dimensional input")
        pca = PCA(n_components=model.pca_components, random_state=model.random_state)
        prepared = pca.fit_transform(prepared).astype(np.float32)

    prepared = cast(NDArray[np.float32], prepared)
    started = time.perf_counter()
    graph = model._build_training_graph(prepared)
    graph_seconds = time.perf_counter() - started
    dataset = UMAPDataset(prepared, graph, random_state=model.random_state)
    if len(dataset) == 0:
        raise ValueError("The training graph has no retained edges to benchmark")
    return prepared, dataset, graph_seconds


def _autocast_context(device: torch.device, config: _PrecisionConfig):
    if config.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=config.autocast_dtype)


def _new_training_state(
    model: ParametricUMAP,
    input_dims: tuple[int, ...],
    initial_state: dict[str, Tensor],
    config: _PrecisionConfig,
):
    module = model._build_model(input_dims)
    module.load_state_dict(initial_state)
    module.to(device=model._device, dtype=config.parameter_dtype)
    optimizer = module.configure_optimizers()
    scaler = GradScaler(
        device=model._device.type,
        enabled=config.use_grad_scaler,
    )
    return module, optimizer, scaler


def _compute_loss(module, batch: tuple[Tensor, Tensor], device: torch.device) -> Tensor:
    edges_to, edges_from = (value.to(device, non_blocking=True) for value in batch)
    parameter_dtype = next(module.parameters()).dtype
    if parameter_dtype != torch.float32:
        edges_to = edges_to.to(parameter_dtype)
        edges_from = edges_from.to(parameter_dtype)
    embedding_to = module.encoder(edges_to)
    embedding_from = module.encoder(edges_from)
    return umap_loss(
        embedding_to,
        embedding_from,
        module._a,
        module._b,
        negative_sample_rate=module.negative_sample_rate,
        repulsion_strength=module.repulsion_strength,
    )


def _optimizer_step(loss: Tensor, optimizer, scaler: GradScaler) -> None:
    optimizer.zero_grad(set_to_none=True)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def _benchmark_candidate(
    candidate: BatchSizeCandidate,
    *,
    model: ParametricUMAP,
    dataset: UMAPDataset,
    input_dims: tuple[int, ...],
    initial_state: dict[str, Tensor],
    config: _PrecisionConfig,
    seed: int,
    epochs: int | None,
    warmup_batches: int = 3,
    timed_batches: int = 20,
) -> None:
    device = model._device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    module, optimizer, scaler = _new_training_state(model, input_dims, initial_state, config)
    module.train()
    dataloader = UMAPDataModule(dataset, candidate.batch_size, model.num_workers).train_dataloader()
    _reset_peak_memory(device)

    try:
        if epochs is None:
            measured_edges = 0
            measured_seconds = 0.0
            effective_warmup = min(warmup_batches, max(candidate.steps_per_epoch - 1, 0))
            iterator = iter(dataloader)
            for index in range(effective_warmup + timed_batches):
                _sync_device(device)
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                with _autocast_context(device, config):
                    loss = _compute_loss(module, batch, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                _optimizer_step(loss, optimizer, scaler)
                _sync_device(device)
                if index >= effective_warmup:
                    measured_seconds += time.perf_counter() - started
                    measured_edges += len(batch[0])
            if measured_edges == 0:
                raise RuntimeError("too few batches to measure after warm-up")
            candidate.edges_per_second = measured_edges / measured_seconds
        else:
            losses: list[float] = []
            throughputs: list[float] = []
            for _ in range(epochs):
                edge_count = 0
                weighted_loss = 0.0
                _sync_device(device)
                started = time.perf_counter()
                for batch in dataloader:
                    with _autocast_context(device, config):
                        loss = _compute_loss(module, batch, device)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    _optimizer_step(loss, optimizer, scaler)
                    batch_edges = len(batch[0])
                    edge_count += batch_edges
                    weighted_loss += float(loss.detach()) * batch_edges
                _sync_device(device)
                elapsed = time.perf_counter() - started
                losses.append(weighted_loss / edge_count)
                throughputs.append(edge_count / elapsed)
            candidate.epoch_losses = tuple(losses)
            measured = throughputs[1:] if len(throughputs) > 1 else throughputs
            candidate.edges_per_second = float(np.median(measured))
        candidate.peak_accelerator_memory_bytes = _peak_memory(device)
        candidate.status = "ok"
    except FloatingPointError as error:
        candidate.status = "non_finite"
        candidate.rejection_reason = str(error)
    except RuntimeError as error:
        if not _is_oom(error):
            raise
        candidate.status = "oom"
        candidate.rejection_reason = "accelerator out of memory"
    finally:
        del module, optimizer, scaler, dataloader
        _release_accelerator_memory(device)


def _apply_quality_gate(
    candidates: list[BatchSizeCandidate],
    reference: BatchSizeCandidate,
    *,
    mode: BenchmarkMode,
    effective_minimum_steps: int,
    loss_tolerance: float,
) -> None:
    if not reference.epoch_losses:
        raise RuntimeError("The reference batch size did not complete its quality run")
    if mode == "quick":
        reference_loss = reference.epoch_losses[-1]
    else:
        reference_loss = float(np.mean(reference.epoch_losses[1:]))

    for candidate in candidates:
        candidate.eligible = False
        if candidate.status != "ok" or not candidate.epoch_losses:
            continue
        if candidate.steps_per_epoch < effective_minimum_steps:
            candidate.rejection_reason = (
                f"{candidate.steps_per_epoch} steps/epoch is below the "
                f"{effective_minimum_steps}-step floor"
            )
            continue
        compared_loss = (
            candidate.epoch_losses[-1]
            if mode == "quick"
            else float(np.mean(candidate.epoch_losses[1:]))
        )
        candidate.loss_ratio = compared_loss / reference_loss
        final_ratio = candidate.epoch_losses[-1] / reference.epoch_losses[-1]
        if candidate.loss_ratio > 1.0 + loss_tolerance or (
            mode == "thorough" and final_ratio > 1.0 + loss_tolerance
        ):
            candidate.rejection_reason = (
                f"loss curve is more than {loss_tolerance:.0%} above the reference"
            )
            continue
        candidate.eligible = True
        candidate.rejection_reason = None


def _choose_recommendation(
    candidates: list[BatchSizeCandidate], throughput_tolerance: float
) -> BatchSizeCandidate:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.eligible and candidate.edges_per_second is not None
    ]
    if not eligible:
        raise RuntimeError("No batch-size candidate passed the benchmark quality gates")
    best_throughput = max(cast(float, candidate.edges_per_second) for candidate in eligible)
    plateau = [
        candidate
        for candidate in eligible
        if cast(float, candidate.edges_per_second) >= best_throughput * (1.0 - throughput_tolerance)
    ]
    return min(plateau, key=lambda candidate: candidate.batch_size)


def recommend_batch_size(
    X: NDArray[np.floating] | Tensor,
    model: ParametricUMAP,
    *,
    mode: BenchmarkMode = "quick",
    batch_sizes: list[int] | tuple[int, ...] | None = None,
    seed: int | None = None,
    minimum_steps_per_epoch: int = 20,
    loss_tolerance: float = 0.05,
    throughput_tolerance: float = 0.05,
) -> BatchSizeRecommendation:
    """Estimate a fast batch size without materially degrading the early loss curve.

    The supplied model is used as configuration and is never fitted or mutated.
    ``quick`` mode performs a short throughput scan followed by two matched
    epochs for the reference and throughput finalists. ``thorough`` mode runs
    five complete epochs for every candidate.
    """
    if mode not in {"quick", "thorough"}:
        raise ValueError("mode must be 'quick' or 'thorough'")
    if minimum_steps_per_epoch <= 0:
        raise ValueError("minimum_steps_per_epoch must be positive")
    if not 0 <= loss_tolerance < 1 or not 0 <= throughput_tolerance < 1:
        raise ValueError("loss and throughput tolerances must be in [0, 1)")

    benchmark_seed = seed if seed is not None else model.random_state
    if benchmark_seed is None:
        benchmark_seed = 42
    benchmark_model = replace(model, random_state=benchmark_seed, extra_callbacks=[])
    benchmark_model.to(model._device)
    precision = _precision_config(benchmark_model.precision, benchmark_model._device)

    torch.manual_seed(benchmark_seed)
    if benchmark_model._device.type == "cuda":
        torch.cuda.manual_seed_all(benchmark_seed)
    prepared, dataset, graph_seconds = _prepare_data(X, benchmark_model)
    graph_edges = len(dataset)
    effective_minimum = min(minimum_steps_per_epoch, graph_edges)

    if batch_sizes is None:
        sizes = _adaptive_batch_sizes(
            graph_edges,
            benchmark_model.batch_size,
            minimum_steps_per_epoch,
            mode,
        )
    else:
        if not batch_sizes or any(size <= 0 for size in batch_sizes):
            raise ValueError("batch_sizes must contain at least one positive integer")
        sizes = sorted(set(batch_sizes))

    candidates = [
        BatchSizeCandidate(
            batch_size=size,
            steps_per_epoch=math.ceil(graph_edges / size),
        )
        for size in sizes
    ]
    eligible_by_steps = [
        candidate for candidate in candidates if candidate.steps_per_epoch >= effective_minimum
    ]
    if not eligible_by_steps:
        raise ValueError(
            "No candidate preserves the minimum optimizer-step floor; include a smaller batch size"
        )

    input_dims = tuple(prepared.shape[1:])
    torch.manual_seed(benchmark_seed)
    initial_module = benchmark_model._build_model(input_dims)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in initial_module.state_dict().items()
    }
    del initial_module
    _release_accelerator_memory(benchmark_model._device)

    if mode == "quick":
        for candidate in candidates:
            _benchmark_candidate(
                candidate,
                model=benchmark_model,
                dataset=dataset,
                input_dims=input_dims,
                initial_state=initial_state,
                config=precision,
                seed=benchmark_seed,
                epochs=None,
            )
        timed = [
            candidate
            for candidate in eligible_by_steps
            if candidate.status == "ok" and candidate.edges_per_second is not None
        ]
        if not timed:
            raise RuntimeError("No batch-size candidate completed the throughput scan")
        reference = min(timed, key=lambda candidate: candidate.batch_size)
        fastest = max(timed, key=lambda candidate: cast(float, candidate.edges_per_second))
        best_rate = cast(float, fastest.edges_per_second)
        plateau = min(
            (
                candidate
                for candidate in timed
                if cast(float, candidate.edges_per_second)
                >= best_rate * (1.0 - throughput_tolerance)
            ),
            key=lambda candidate: candidate.batch_size,
        )
        finalists = {reference.batch_size, fastest.batch_size, plateau.batch_size}
        for candidate in candidates:
            if candidate.batch_size not in finalists:
                if candidate.status == "ok":
                    candidate.status = "timing_only"
                    candidate.rejection_reason = "not selected for the quick quality run"
                continue
            _benchmark_candidate(
                candidate,
                model=benchmark_model,
                dataset=dataset,
                input_dims=input_dims,
                initial_state=initial_state,
                config=precision,
                seed=benchmark_seed,
                epochs=2,
            )
    else:
        for candidate in candidates:
            _benchmark_candidate(
                candidate,
                model=benchmark_model,
                dataset=dataset,
                input_dims=input_dims,
                initial_state=initial_state,
                config=precision,
                seed=benchmark_seed,
                epochs=5,
            )
        completed = [candidate for candidate in candidates if candidate.status == "ok"]
        if not completed:
            raise RuntimeError("No batch-size candidate completed the quality run")
        reference = min(completed, key=lambda candidate: candidate.batch_size)

    _apply_quality_gate(
        candidates,
        reference,
        mode=mode,
        effective_minimum_steps=effective_minimum,
        loss_tolerance=loss_tolerance,
    )
    chosen = _choose_recommendation(candidates, throughput_tolerance)
    return BatchSizeRecommendation(
        recommended_batch_size=chosen.batch_size,
        mode=mode,
        device=str(benchmark_model._device),
        precision=benchmark_model.precision,
        seed=benchmark_seed,
        graph_edges=graph_edges,
        graph_build_seconds=graph_seconds,
        minimum_steps_per_epoch=effective_minimum,
        loss_tolerance=loss_tolerance,
        throughput_tolerance=throughput_tolerance,
        candidates=tuple(candidates),
    )
