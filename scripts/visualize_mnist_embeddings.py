from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from glass_box_umap import GlassBoxUMAP

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

# reducer = UMAP()
# reducer.fit(X)
# output = reducer.transform(X)

# plt.scatter(output[:, 0], output[:, 1], c=colors, s=5)
# plt.savefig("umap_original.png")
# plt.close()

for epoch in [8]:  # 1, 5, 10, 50, 100, 200, 500]:
    reducer = GlassBoxUMAP(
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
    plt.savefig(f"mar24_pr_umap_glassbox_{epoch}.png")
    plt.close()

    # ── Compute Jacobians ─────────────────────────────────────────────────────
    print("Computing Jacobians...")
    encoder = reducer._fitted_model.encoder
    X_tensor = torch.tensor(X, device=reducer._device)

    encoder.eval()
    with torch.no_grad():
        Z_np = encoder(X_tensor).cpu().numpy()

    encoder_for_jac = reducer.prelu_to_leaky(encoder)
    J = reducer.compute_jacobian(encoder_for_jac, X_tensor)
    J_np = J.cpu().numpy()
    # X = X_tensor.cpu().numpy()

    # ── Verify Jacobian exactness ─────────────────────────────────────────────
    reducer.verify_jacobian(Z_np, J_np, X)

    # Feature importance: L2 norm over embedding dims -> (n, n_dims)
    feat_importance = np.linalg.norm(np.einsum("noi,ni->noi", J_np, X), axis=1)  # (n, n_dims)

    labels = y
    elem_prod = np.einsum("noi,ni->noi", J_np, X)
    for cluster_id in range(10):
        # mask = y==str(cluster_id)
        cluster_mask = labels == cluster_id
        # mean_imp = np.median(feat_importance[cluster_mask],axis=0)

        mean_imp = np.mean(elem_prod[cluster_mask][:, 0, :] ** 2, axis=0)

        mean_imp2 = np.mean(elem_prod[cluster_mask][:, 1, :] ** 2, axis=0)
        top_genes = np.argsort(mean_imp)[::-1][:20]
        print(f"Cluster {cluster_id} top genes: {top_genes}")
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(mean_imp.reshape([28, 28]))
        plt.subplot(1, 2, 2)
        plt.imshow(mean_imp2.reshape([28, 28]))
        plt.savefig(f"mar24_mnist_cluster_{cluster_id}.png")
