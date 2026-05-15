import threading
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CategoricalColorMapper,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LinearScale,
    LogScale,
    Slider,
    Span,
    Toggle,
)
from bokeh.models.layouts import Column
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.server.server import Server
from bokeh.util.warnings import BokehUserWarning
from numpy.typing import NDArray
from tornado.ioloop import IOLoop

from ._colors import pick_palette
from ._hover import _HOVER_IMAGE_KEY, _to_hover_uri
from ._scatter import OutputBackend

warnings.filterwarnings("ignore", category=BokehUserWarning, message="reference already known")

LOSS_PLOT_HEIGHT = 200
PLAY_BUTTON_WIDTH = 80
LOG_BUTTON_WIDTH = 80
SAVE_BUTTON_WIDTH = 120


class LiveEmbeddingCallback(pl.Callback):
    """Pytorch Lightning callback that serves a live-updating Bokeh scatter.

    Spins up a Bokeh server on a background thread, opens a browser tab, and
    streams a fresh embedding (via ``transform_fn``) to the page after each
    training epoch starts. Each session keeps a per-frame history that the
    user can scrub through with a slider, play back with a button, or export
    to a self-contained HTML file. Training keeps running on the main thread;
    updates cross to the Bokeh event loop via
    ``Document.add_next_tick_callback``.

    Args:
        transform_fn: Callable that maps the high-dimensional ``X`` to a
            ``(n_samples, 2)`` array. Typically the embedder's ``transform``
            method.
        X: High-dimensional input fed to ``transform_fn`` after each epoch.
        labels: Optional per-sample categorical labels for coloring.
        port: Port the Bokeh server listens on. ``0`` (default) lets the OS
            pick a free port, which avoids ``EADDRINUSE`` collisions when the
            callback is re-instantiated within the same process (e.g. a Jupyter
            kernel that already hosts a previous run's server).
        output_backend: Bokeh rendering backend for the scatter. Defaults to
            ``"webgl"``; switch to ``"canvas"`` if the GPU/driver/browser
            combination renders the plot incorrectly.
        hover_images: Optional uint8 image array of shape
            ``(n_samples, H, W)`` or ``(n_samples, H, W, 3 | 4)``. When set,
            each tooltip shows the sample's image above the index/label text.
        block_after_fit: When ``True`` (default), block at the end of training
            so the Bokeh server keeps serving until the user presses Ctrl-C.
            Set to ``False`` from interactive contexts (e.g. Jupyter) where
            the host process already keeps the server alive.
    """

    def __init__(
        self,
        transform_fn: Callable[[torch.Tensor], NDArray[np.floating]],
        X: torch.Tensor,
        labels: list[str] | None = None,
        port: int = 0,
        output_backend: OutputBackend = "webgl",
        hover_images: NDArray[np.uint8] | None = None,
        block_after_fit: bool = True,
    ) -> None:
        self.transform_fn = transform_fn
        self.X = X
        self.labels = labels
        self.factors = sorted(set(labels)) if labels is not None else None
        self.port = port
        self.output_backend = output_backend
        self.hover_images = hover_images
        self.block_after_fit = block_after_fit
        self._hover_uris: list[str] | None = (
            [_to_hover_uri(img) for img in hover_images] if hover_images is not None else None
        )
        self.frames: list[NDArray[np.floating]] = []
        self.losses: list[float] = []
        self.latest_epoch = -1
        self.sessions: list[tuple[Document, ColumnDataSource, ColumnDataSource]] = []
        self._ready = threading.Event()
        self._server_error: BaseException | None = None

    def _tooltip(self) -> str:
        parts: list[str] = []
        if self._hover_uris is not None:
            parts.append(
                f"<img src='@{_HOVER_IMAGE_KEY}' style='display:block; margin-bottom:4px'/>"
            )
        body = "index: @index"
        if self.labels is not None:
            body += " &nbsp;&middot;&nbsp; label: @label"
        parts.append(body)
        return f"<div>{''.join(parts)}</div>"

    def _build_layout(
        self,
        frames: list[NDArray[np.floating]],
        losses: list[float],
        is_live: bool,
    ) -> tuple[Column, ColumnDataSource, ColumnDataSource]:
        n_pts = self.X.shape[0]
        if frames:
            initial = frames[-1]
            initial_epoch = len(frames) - 1
        else:
            initial = np.zeros((n_pts, 2), dtype=np.float32)
            initial_epoch = -1

        source_data = dict(
            x=initial[:, 0].tolist(),
            y=initial[:, 1].tolist(),
            index=list(range(n_pts)),
        )
        if self.labels is not None:
            source_data["label"] = self.labels
        if self._hover_uris is not None:
            source_data[_HOVER_IMAGE_KEY] = list(self._hover_uris)
        display_source = ColumnDataSource(source_data)

        frames_source = ColumnDataSource(
            dict(
                x=[f[:, 0].tolist() for f in frames],
                y=[f[:, 1].tolist() for f in frames],
            )
        )

        title_text = "Initializing" if initial_epoch < 0 else f"Epoch {initial_epoch}"
        p = figure(
            title=title_text,
            sizing_mode="stretch_both",
            output_backend=self.output_backend,  # type: ignore[arg-type]
        )

        p.add_tools(HoverTool(tooltips=self._tooltip()))
        if self.labels is not None and self.factors is not None:
            color_mapping = CategoricalColorMapper(
                factors=self.factors, palette=pick_palette(len(self.factors))
            )
            p.scatter(
                "x",
                "y",
                source=display_source,
                color={"field": "label", "transform": color_mapping},
                size=4,
            )
        else:
            p.scatter("x", "y", source=display_source, color="navy", size=4)

        loss_source = ColumnDataSource(dict(epoch=list(range(len(losses))), loss=list(losses)))
        loss_plot = figure(
            sizing_mode="stretch_width",
            height=LOSS_PLOT_HEIGHT,
            x_axis_label="epoch",
            y_axis_label="loss",
            toolbar_location=None,
        )
        loss_plot.line(x="epoch", y="loss", source=loss_source, line_width=2)
        loss_plot.add_tools(
            HoverTool(
                tooltips=[("epoch", "@epoch"), ("loss", "@loss{0.0000}")],
                mode="vline",
            )
        )

        epoch_span = Span(
            location=initial_epoch if initial_epoch >= 0 else 0,
            dimension="height",
            line_color="red",
            line_dash="dashed",
            line_width=1,
        )
        loss_plot.add_layout(epoch_span)

        initial_end = max(1, len(frames) - 1)
        slider = Slider(
            start=0,
            end=initial_end,
            value=initial_end,
            step=1,
            title="epoch",
            sizing_mode="stretch_width",
        )
        play_button = Button(label="Play", width=PLAY_BUTTON_WIDTH)

        log_toggle = Toggle(label="Log Y", width=LOG_BUTTON_WIDTH)
        linear_scale = LinearScale()
        log_scale = LogScale()
        log_toggle.js_on_change(
            "active",
            CustomJS(
                args=dict(
                    loss_plot=loss_plot,
                    linear_scale=linear_scale,
                    log_scale=log_scale,
                ),
                code="""
                    loss_plot.y_scale = cb_obj.active ? log_scale : linear_scale;
                """,
            ),
        )

        scrub_callback = CustomJS(
            args=dict(
                display_source=display_source,
                frames_source=frames_source,
                title=p.title,
                epoch_span=epoch_span,
            ),
            code="""
                const i = cb_obj.value;
                const x = frames_source.data.x[i];
                const y = frames_source.data.y[i];
                if (x === undefined || y === undefined) return;
                display_source.data['x'] = x;
                display_source.data['y'] = y;
                display_source.change.emit();
                title.text = `Epoch ${i}`;
                epoch_span.location = i;
            """,
        )
        slider.js_on_change("value", scrub_callback)

        play_callback = CustomJS(
            args=dict(slider=slider),
            code="""
                if (window._play_interval) {
                    clearInterval(window._play_interval);
                    window._play_interval = null;
                    cb_obj.label = 'Play';
                    return;
                }
                if (slider.value >= slider.end) {
                    slider.value = slider.start;
                }
                cb_obj.label = 'Pause';
                window._play_interval = setInterval(() => {
                    if (slider.value >= slider.end) {
                        clearInterval(window._play_interval);
                        window._play_interval = null;
                        cb_obj.label = 'Play';
                        return;
                    }
                    slider.value = slider.value + 1;
                }, 50);
            """,
        )
        play_button.js_on_click(play_callback)

        controls: list = [log_toggle, play_button, slider]
        if is_live:
            new_frame_callback = CustomJS(
                args=dict(
                    slider=slider,
                    frames_source=frames_source,
                    display_source=display_source,
                    title=p.title,
                    epoch_span=epoch_span,
                ),
                code="""
                    const n = frames_source.data.x.length;
                    if (n === 0) return;
                    const new_end = Math.max(1, n - 1);
                    const was_at_end = slider.value === slider.end;
                    slider.end = new_end;
                    if (was_at_end) {
                        slider.value = new_end;
                        const idx = Math.min(new_end, n - 1);
                        display_source.data['x'] = frames_source.data.x[idx];
                        display_source.data['y'] = frames_source.data.y[idx];
                        display_source.change.emit();
                        title.text = `Epoch ${idx}`;
                        epoch_span.location = idx;
                    }
                """,
            )
            frames_source.js_on_change("data", new_frame_callback)

            save_button = Button(label="Save HTML", width=SAVE_BUTTON_WIDTH)
            save_button.on_click(self._save_html)
            controls.append(save_button)

        layout = column(
            p,
            loss_plot,
            row(*controls, sizing_mode="stretch_width"),
            sizing_mode="stretch_both",
            styles={
                "max-width": "900px",
                "min-height": "600px",
                "max-height": "1100px",
            },
        )
        return layout, frames_source, loss_source

    def _save_html(self) -> None:
        frames_snapshot = list(self.frames)
        losses_snapshot = list(self.losses)
        layout, _, _ = self._build_layout(frames_snapshot, losses_snapshot, is_live=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(f"glassbox_live_{timestamp}.html").resolve()
        html = file_html(layout, INLINE, title="Glass Box UMAP")
        filename.write_text(html)
        print(f"Saved {filename}")

    def _doc_handler(self, doc: Document) -> None:
        layout, frames_source, loss_source = self._build_layout(
            self.frames, self.losses, is_live=True
        )
        doc.add_root(layout)

        entry = (doc, frames_source, loss_source)
        self.sessions.append(entry)

        def cleanup(_ctx: object) -> None:
            if entry in self.sessions:
                self.sessions.remove(entry)

        doc.on_session_destroyed(cleanup)

    def _start_server(self) -> None:
        try:
            io_loop = IOLoop()
            server = Server(
                {"/": Application(FunctionHandler(self._doc_handler))},
                io_loop=io_loop,
                port=self.port,
                session_token_expiration=86400,
            )
            self.port = server.port
            server.start()
            io_loop.add_callback(server.show, "/")
            io_loop.add_callback(self._ready.set)
            io_loop.start()
        except BaseException as exc:
            self._server_error = exc
            self._ready.set()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        threading.Thread(target=self._start_server, daemon=True).start()
        self._ready.wait(timeout=5.0)
        if self._server_error is not None:
            raise self._server_error
        print(f"Live embedding serving at http://localhost:{self.port}/")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        emb = self.transform_fn(self.X)
        self.frames.append(emb)
        self.latest_epoch = trainer.current_epoch

        new_x = emb[:, 0].tolist()
        new_y = emb[:, 1].tolist()

        for doc, frames_source, _loss_source in list(self.sessions):

            def update(frames_source=frames_source, new_x=new_x, new_y=new_y) -> None:
                frames_source.data = {
                    "x": [*frames_source.data["x"], new_x],
                    "y": [*frames_source.data["y"], new_y],
                }

            doc.add_next_tick_callback(update)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        loss = trainer.callback_metrics.get("loss_epoch")
        if loss is None:
            return
        new_loss = float(loss)
        new_epoch = trainer.current_epoch
        self.losses.append(new_loss)

        for doc, _frames_source, loss_source in list(self.sessions):

            def update(loss_source=loss_source, new_loss=new_loss, new_epoch=new_epoch) -> None:
                loss_source.data = {
                    "epoch": [*loss_source.data["epoch"], new_epoch],
                    "loss": [*loss_source.data["loss"], new_loss],
                }

            doc.add_next_tick_callback(update)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.block_after_fit:
            print(f"Training done. Server still serving at http://localhost:{self.port}/.")
            return
        print(
            f"Training done. Server still serving at http://localhost:{self.port}/. "
            "Press Ctrl-C to exit."
        )
        threading.Event().wait()
