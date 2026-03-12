from pathlib import Path

import matplotlib.pyplot as plt
import torch
from glass_box_umap import GlassBoxUMAP, ParametricUMAP
from umap import UMAP

X = torch.load("./tests/fixtures/mnist_images.pt")
y = torch.load("./tests/fixtures/mnist_labels.pt")

color_dict = {
    0: "red",
    1: "green",
    2: "blue",
    3: "black",
    4: "magenta",
    5: "cyan",
    6: "orange",
    7: "salmon",
    8: "violet",
    9: "purple",
}
colors = [color_dict.get(x.item(), "grey") for x in y]

reducer = UMAP()
reducer.fit(X)
output = reducer.transform(X)

plt.scatter(output[:, 0], output[:, 1], c=colors, s=5)
plt.savefig("umap_original.png")
plt.close()

for epoch in [4]:#1, 5, 10, 50, 100, 200, 500]:
    reducer = ParametricUMAP(
        epochs=epoch,
        lr=1e-3,
        batch_size=256,
        repulsion_strength=1.0,
        encoder_name="default",
        checkpoint_dir=Path(f"tmp_{epoch}"),
    )
    # reducer = GlassBoxUMAP(
    #    epochs=epoch,
    #    lr=1e-3,
    #    batch_size=256,
    #    repulsion_strength=1.0,
    #    encoder_kwargs={"hidden_size": 1024},
    #    checkpoint_dir=Path(f"tmp_{epoch}"),
    # )

    reducer.fit(X)
    output = reducer.transform(X)

    plt.scatter(output[:, 0], output[:, 1], c=colors, s=5)
    plt.savefig(f"feb_26_pr_umap_glassbox_{epoch}.png")
    plt.close()
