pcaout = np.einsum('ijk,ik->ij', np.array(jacobian), adata.obsm["X_pca"])#.shape
attribution_matrix = np.einsum('ijk,ik->ijk', np.array(jacobian), adata.obsm["X_pca"])#.reshape([jacobxall.shape[0],-1])[:500,:]
attribution_matrix = attribution_matrix.reshape([jacobxall.shape[0],-1])
# Step 2: Cluster on attribution patterns
# from sklearn.cluster import DBSCAN, KMeans
# clusters = DBSCAN(eps=2.5).fit_predict(attribution_matrix)

from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.metrics import silhouette_score

# Try KMeans with higher k than expected cell types
k = 100  # Oversegment intentionally
# kmeans = KMeans(n_clusters=k, n_init=10)
# clusters = kmeans.fit_predict(attribution_pcs)

# Or hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=40)
clusters = hierarchical.fit_predict(attribution_matrix)
clusters
plt.figure()
plt.scatter(pcaout[:,0],pcaout[:,1],s=0.4,c=clusters,cmap='tab20')
plt.title("Agg Clustering by Features")
plt.figure()
plt.scatter(pcaout[:,0],pcaout[:,1],s=0.4,c=adata.obs['cell_type'].cat.codes,cmap='tab20')
plt.title("Cell type")
plt.figure()
plt.scatter(pcaout[:,0],pcaout[:,1],s=0.4,c=adata.obs['leiden'].cat.codes,cmap='tab20')
plt.title("Leiden")
cttop = adata.obs["cell_type"].value_counts().index[2]
indices_ci = [ci for ci,ct in enumerate(adata.obs["cell_type"]) if ct in cttop]
indices_ct = [ct for ct in adata.obs["cell_type"] if ct in cttop]
len(indices_ci)
import seaborn as sns

# Create contingency table
contingency = pd.crosstab(adata.obs['cell_type'][indices_ci], clusters[indices_ci])

# Normalize by cell type to see distribution
contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)

# Visualize
plt.figure(figsize=(20, 10))
sns.heatmap(contingency_norm, cmap='Blues', cbar_kws={'label': 'Fraction of cells'})
plt.xlabel('Attribution Cluster')
plt.ylabel('Cell Type')
plt.title('Cell Type Distribution Across Attribution Clusters')

# Find cell types split across multiple attribution clusters
split_cell_types = contingency_norm.index[
    (contingency_norm > 0.1).sum(axis=1) > 1  # Present in >1 cluster at >10%
]
print(f"Cell types with multiple attribution patterns: {split_cell_types.tolist()}")
attribution_clusters=clusters
for cell_type in split_cell_types:
    # Get cells of this type
    mask = adata.obs['cell_type'] == cell_type
    cells_of_type = np.where(mask)[0]
    
    # Get their attribution clusters
    clusters_for_type = attribution_clusters[mask]
    unique_clusters = np.unique(clusters_for_type)
    
    print(f"\n{cell_type} spans {len(unique_clusters)} attribution clusters")
    
    # For each attribution pattern in this cell type
    for cluster in unique_clusters:
        cluster_mask = attribution_clusters[mask] == cluster
        n_cells = cluster_mask.sum()
        print(f"  Cluster {cluster}: {n_cells} cells ({100*n_cells/len(cells_of_type):.1f}%)")
genes
def compare_attribution_patterns(cell_type, attribution_matrix, cell_types, attribution_clusters, n_genes=8, gene_names=genes, showPlot=False):
    mask = cell_types == cell_type
    
    # Get unique attribution clusters for this cell type
    clusters_for_type = np.unique(attribution_clusters[mask])
    
    if showPlot:
        fig, axes = plt.subplots(len(clusters_for_type), 1, figsize=(12, 4*len(clusters_for_type)))
        if len(clusters_for_type) == 1:
            axes = [axes]
    
    # Store mean attributions for comparison
    cluster_attributions = {}
    
    for idx, cluster in enumerate(clusters_for_type):
        # Get cells of this type in this attribution cluster
        cluster_mask = (cell_types == cell_type) & (attribution_clusters == cluster)
        mean_attr = attribution_matrix[cluster_mask].mean(axis=0)
        
        # Separate UMAP1 and UMAP2 contributions
        attr_umap1 = mean_attr[:n_genes]
        attr_umap2 = mean_attr[n_genes:]
        
        # Find top contributing genes
        top_genes_idx = np.argsort(np.abs(mean_attr))[-20:]
        
        cluster_attributions[cluster] = {
            'mean': mean_attr,
            'top_genes': gene_names[top_genes_idx % n_genes],
            'n_cells': cluster_mask.sum()
        }
        
        if showPlot and cluster_mask.sum()>10:
            # Plot
            axes[idx].bar(range(20), mean_attr[top_genes_idx])
            axes[idx].set_xticks(range(20))
            axes[idx].set_xticklabels(gene_names[top_genes_idx % n_genes], rotation=45, ha='right')
            axes[idx].set_title(f'{cell_type} - Attribution Pattern {cluster} ({cluster_mask.sum()} cells)')
            axes[idx].set_ylabel('Mean Attribution')
    if showPlot:
        plt.tight_layout()
    return cluster_attributions

# Analyze each split cell type
for cell_type in split_cell_types[:5]:  # Top 5 for visualization
    patterns = compare_attribution_patterns(cell_type, attribution_matrix, adata.obs['cell_type'], attribution_clusters, showPlot=True)
def differential_attribution(cell_type, cluster1, cluster2, attribution_matrix):
    """Compare attribution patterns between two clusters of same cell type"""
    
    mask1 = (cell_types == cell_type) & (attribution_clusters == cluster1)
    mask2 = (cell_types == cell_type) & (attribution_clusters == cluster2)
    
    attr1 = attribution_matrix[mask1].mean(axis=0)
    attr2 = attribution_matrix[mask2].mean(axis=0)
    
    # Compute difference
    diff = attr1 - attr2
    
    # Find genes with largest attribution differences
    diff_genes_idx = np.argsort(np.abs(diff))[-30:]
    
    # Create DataFrame for easy viewing
    results = pd.DataFrame({
        'gene': gene_names[diff_genes_idx % n_genes],
        'umap_dim': ['UMAP1' if idx < n_genes else 'UMAP2' for idx in diff_genes_idx],
        'attr_cluster1': attr1[diff_genes_idx],
        'attr_cluster2': attr2[diff_genes_idx],
        'difference': diff[diff_genes_idx]
    })
    
    return results.sort_values('difference', key=abs, ascending=False)


from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# hierarchical = AgglomerativeClustering(n_clusters=40)
# clusters = hierarchical.fit_predict(attribution_matrix)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=20, n_clusters=None)

model = model.fit(attribution_matrix)
plot_dendrogram(model, truncate_mode="level", p=4)


from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Perform hierarchical clustering
linkage_matrix = linkage(attribution_matrix, method='ward')

# Get cluster assignments at each merge
n_cells = len(attribution_matrix)
cluster_labels = np.arange(40)

# First, get hierarchical clusters
n_clusters = 40  # Or whatever makes sense
clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Train decision tree to predict clusters from attributions
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)
dt.fit(attribution_matrix, clusters)

# Visualize with gene names
plt.figure(figsize=(20, 10))
plot_tree(dt, 
          feature_names=[f"{gene}_{dim}" for gene in genes for dim in ['UMAP1', 'UMAP2']], 
          class_names=[f"Cluster_{i}" for i in range(n_clusters)],
          filled=True,
          max_depth=3,  # Show only top splits
          fontsize=10)
plt.title("Attribution Decision Tree")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dendrogram(linkage_matrix, attribution_matrix, gene_names):
    # Create dendrogram
    dendro = dendrogram(linkage_matrix, no_plot=True)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add dendrogram lines
    for i, (x, y) in enumerate(zip(dendro['icoord'], dendro['dcoord'])):
        # Find genes for this split
        merge_idx = i
        if merge_idx < len(linkage_matrix):
            # Get split info
            left_idx = int(linkage_matrix[merge_idx, 0])
            right_idx = int(linkage_matrix[merge_idx, 1])
            
            # Simple heuristic for top genes
            # (In practice, you'd compute actual attributions)
            gene_info = f"Split {i}: Gene1, Gene2, Gene3"
        else:
            gene_info = "Leaf"
            
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='blue', width=1),
            hovertext=gene_info,
            hoverinfo='text',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Interactive Attribution Dendrogram",
        xaxis_title="Cell Index",
        yaxis_title="Distance",
        hovermode='closest'
    )
    
    return fig
fig = create_interactive_dendrogram(linkage_matrix, attribution_matrix, genes)
fig.show()