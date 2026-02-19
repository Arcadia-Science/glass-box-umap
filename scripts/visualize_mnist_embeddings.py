from pathlib import Path

import matplotlib.pyplot as plt
import time
import torch
from glass_box_umap import GlassBoxUMAP, ParametricUMAP
from umap import UMAP

from matplotlib.markers import MarkerStyle

if __name__ == "__main__":
    X = torch.load("./tests/fixtures/mnist_images.pt")/255.
    X -= X.mean(axis=0)
    # X /= X.std(axis=0)
    # std = X.std(axis=0)
    # X[:,std!=0] = X[:,std!=0]/std[std!=0]
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

    # reducer = UMAP()
    # reducer.fit(X)
    # output = reducer.transform(X)

    # plt.scatter(output[:, 0], output[:, 1], c=colors, s=5)
    # plt.savefig("umap_original.png")
    # plt.close()
    st = time.time()
    # desc="old_ds"
    desc="n_neigh_25_md_2_n_epochs_2_batch128_8_rep1_newdata_rs1323_adamw_rr"
    for epoch in [8]:
        reducer = ParametricUMAP(
            epochs=epoch,
            lr=1e-3,
            batch_size=128*4*2,
            repulsion_strength=1.0,
            min_dist=0.02,
            n_neighbors=25,
            encoder_name="default",
            checkpoint_dir=Path(f"tmp_{epoch}"),
            random_state=1323,
            num_workers=10
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
    plt.savefig(f"umap_glassbox_{epoch}_{desc}.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        output[:, 0], 
        output[:, 1], 
        c=colors, 
        cmap='Spectral', 
        s=.1, 
        # marker='.',
        marker=MarkerStyle('o', fillstyle='full'),
        # fillstyle='filled',
        alpha=0.5,
        rasterized=True,
    )
    plt.colorbar(scatter, label='Digit Label')
    plt.title("Parametric UMAP with FC Display Settings", fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"fc_umap_glassbox_{epoch}_{desc}.png", dpi=300, bbox_inches='tight')
    plt.close()
    # print(f"Saved {filename}")

    # fig, ax = plt.subplots(figsize=(10,8))
    print("Elapsed time: ", (st-time.time())/60.)

    # 2. Plot with optimized settings
    # ax.scatter(
    #     output[:, 0], 
    #     output[:, 1], 
    #     s=0.1,
    #     c=colors,             # Small marker size
    #     alpha=0.1,        # Low opacity to show density
    #     edgecolors='none', # Remove borders to save memory/space
    #     marker='.',       # Use the smallest point marker
    #     rasterized=True   # IMPORTANT: Keeps PDF file size small and fast to open
    # )

    # ax.set_title("Density Distribution of 100,000 Points")
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")

    # # 3. Export as PDF
    # plt.savefig(f"opt_umap_glassbox_{epoch}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()