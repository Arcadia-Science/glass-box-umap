from pathlib import Path

import matplotlib.pyplot as plt
import time
import torch
from glass_box_umap import GlassBoxUMAP, ParametricUMAP
from umap import UMAP

from matplotlib.markers import MarkerStyle

if __name__ == "__main__":
    
        #@title glass box UMAP
    import os
    import anndata as ad
    import numpy as np
    import pandas as pd
    from matplotlib import font_manager as fm, pyplot as plt

    # Config
    TRAIN = False

    N_FITS = 1
    N_FITS_TO_LOAD = 1
    N_PCS = 50
    EPOCHS = 64
    RANDOM_STATE = 42
    GROUPBY_KEY = 'cell_type'
    MODEL_PATH_PATTERN = "models/umap_{i}.pth"
    SUMMARY_BASENAME = "saved_outputs/bmmc_features_rev"
    BATCH_KEY = "Samplename"

    summary_stats_file = f"{SUMMARY_BASENAME}_stats.csv"
    summary_plot_file = f"{SUMMARY_BASENAME}_plot_data.npz"
    summary_interactive_file = f"{SUMMARY_BASENAME}_interactive.csv"

    # %%
    # | code-summary: "Data and preprocessing methods"
    # | code-fold: true
    # | output: false

    import os
    import subprocess
    import scanpy as sc
    import anndata as ad
    import pandas as pd

    def download_bone_marrow_data(
        url="ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz",
        filename="GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
    ) -> ad.AnnData:
        """
        Downloads, unzips, and loads the bone marrow dataset.
        """
        unzipped_filename = filename.replace(".gz", "")
        if not os.path.isfile(unzipped_filename):
            if not os.path.isfile(filename):
                subprocess.run(["wget", url, "--no-verbose"])
            subprocess.run(["gunzip", filename])

        return sc.read_h5ad(unzipped_filename)

    def preprocess_adata(
        adata: ad.AnnData,
        min_genes: int = 100,
        min_cells: int = 3,
        n_top_genes: int = 2000,
        n_pcs: int = 50,
        batch_key: str = "Samplename",
        run_scrublet: bool = False
    ) -> ad.AnnData:
        """
        Runs the full scRNA-seq preprocessing pipeline on an AnnData object.

        Args:
            adata (ad.AnnData): The raw AnnData object.
            min_genes (int): Min genes for cell filtering.
            min_cells (int): Min cells for gene filtering.
            n_top_genes (int): Number of highly variable genes to select.
            n_pcs (int): Number of principal components to compute.
            batch_key (str): The key in .obs for batch correction (if any).
            run_scrublet (bool): Whether to run doublet detection.

        Returns:
            ad.AnnData: The processed AnnData object.
        """
        print("--- Starting Preprocessing ---")

        # 1. Initial setup and QC gene flagging
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
        adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

        # 2. Calculate QC
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

        # 3. Remove QC genes
        genes_to_remove = adata.var["mt"] | adata.var["ribo"]
        adata._inplace_subset_var(~genes_to_remove)

        # 4. Filter cells, genes, and detect doublets
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        if run_scrublet:
            sc.pp.scrublet(adata, batch_key=batch_key)

        # 5. Normalize and find HVGs
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)

        # 6. Run PCA
        sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)

        return adata

    # %% [markdown]
    # The preprocessing pipeline follows the standard procedure for the dataset in [the ScanPy (v1.11.4) clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html). We take the extra step of filtering out the less common cell types to simplify visualizations and keep only the top 12.

    # %%
    # | code-summary: "Load and preprocess data"
    # | code-fold: true
    # | output: false

    def prepare_data(groupby_key: str = 'cell_type', n_pcs: int = 50, batch_key: str = "Samplename") -> ad.AnnData:
        """
        Loads, concatenates, and preprocesses the scRNA-seq data.
        """
        adata_raw = download_bone_marrow_data()
        adata_raw.var_names_make_unique()

        # Slice and concatenate data
        if groupby_key == 'cell_type':
            adata_filtered = adata_raw
            # adata_subset1 = adata_raw[adata_raw.obs['Samplename'] == 'site1_donor1_cite', :].copy()
            # adata_subset3 = adata_raw[adata_raw.obs['Samplename'] == 'site1_donor3_cite', :].copy()
            # adata_filtered = ad.concat([adata_subset1, adata_subset3], label="donors")
        else:
            adata_filtered = adata_raw

        adata_filtered.obs_names_make_unique()

        # Run the preprocessing pipeline on ALL concatenated cells first
        adata_processed = preprocess_adata(
            adata_filtered,  # <-- Use the unfiltered data
            n_top_genes=2000,
            # n_top_genes=384*2,
            n_pcs=n_pcs,
            batch_key=batch_key
        )

        # Now, filter to the top cell types for your analysis

        if groupby_key == 'cell_type':
            top_cell_types = adata_processed.obs[groupby_key].value_counts().nlargest(12).index
            # This subset now contains the .obsm['X_pca'] that was calculated on ALL cells
            adata_final = adata_processed[adata_processed.obs[groupby_key].isin(top_cell_types)].copy()
        if groupby_key.lower() == 'cd4+_t_cell_type':
            adata_subset1 = adata_processed[adata_processed.obs['cell_type'] == 'CD4+ T activated', :].copy()
            adata_subset3 = adata_processed[adata_processed.obs['cell_type'] == 'CD4+ T naive', :].copy()

            adata_t_cells = ad.concat([adata_subset1,adata_subset3], label="T_cell_type")

            adata_final = scp.preprocess_adata(
                adata_t_cells,
                n_top_genes=100, # You can use fewer HVGs for a subset
                n_pcs=50,         # You need fewer PCs for a subset
                batch_key="Samplename"
            )

        return adata_final

    N_FITS = 1
    N_FITS_TO_LOAD = 1
    N_PCS = 50
    TRAIN = True
    EPOCHS = 64
    RANDOM_STATE = 42
    GROUPBY_KEY = 'cell_type'
    MODEL_PATH_PATTERN = "models/umap_{i}.pth"
    SUMMARY_BASENAME = "saved_outputs/bmmc_features_rev"
    BATCH_KEY = "Samplename"

    summary_stats_file = f"{SUMMARY_BASENAME}_stats.csv"
    summary_plot_file = f"{SUMMARY_BASENAME}_plot_data.npz"
    summary_interactive_file = f"{SUMMARY_BASENAME}_interactive.csv"

    adata = prepare_data(
        groupby_key=GROUPBY_KEY,
        n_pcs=N_PCS,
        batch_key=BATCH_KEY
    )

    # reducer = UMAP()
    # reducer.fit(X)
    # output = reducer.transform(X)

    # plt.scatter(output[:, 0], output[:, 1], c=colors, s=5)
    # plt.savefig("umap_original.png")
    # plt.close()
    st = time.time()
    # desc="old_ds"
    # desc="n_neigh_25_md_2_n_epochs_2_batch128_8_rep1_newdata_rs1323_adamw_30K"
    desc="gene_exp_md_05_batch_256_x_4_lr1em4_rep12_pca"
    # desc="gene_exp_md_05_batch_256_x_4_lr1em4_rep12_hvg100"
    for epoch in [4]:
        reducer = ParametricUMAP(
            epochs=epoch,
            lr=1e-4,
            batch_size=128*1*2*4,
            repulsion_strength=18.0,
            min_dist=0.5,
            n_neighbors=10,
            encoder_name="default",
            checkpoint_dir=Path(f"tmp_{epoch}"),
            random_state=11,
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

    reducer.fit(adata.obsm['X_pca'])
    output = reducer.transform(adata.obsm['X_pca'])
    
    # print("X shape: ", adata.obsm['X'].shape)

    # adata_hvg = adata[:, adata.var['highly_variable']].X.toarray()
    # print("X shape: ", adata_hvg.shape)
    # reducer.fit(adata_hvg)
    # output = reducer.transform(adata_hvg)

    plt.scatter(output[:, 0], output[:, 1], c=adata.obs['cell_type'].astype('category').cat.codes, s=2., cmap='tab20')
    plt.savefig(f"umap_glassbox_{epoch}_{desc}.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        output[:, 0], 
        output[:, 1], 
        c=adata.obs['cell_type'].astype('category').cat.codes, 
        cmap='tab20', 
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