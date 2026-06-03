import os
import numpy as np
import pandas as pd
import scyan
import torch
import random
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from minisom import MiniSom
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import time


# Scyan

def run_scyan(
    adata,
    knowledgetable,
    knowledgetable_dir,
    base_dir,
    unassigned_label,
    phenotype_level="region",
    filter=None,              
    filter_for=None,    
    max_area=None,  
    batch_key=None,  
    continuous_covariates=None,     
    prior_std=0.3,
    lr=0.0005,
    log_prob_th=-100,
    max_epochs=400,
    patience=8,
    min_delta=0.1,
    polarity=False,
    random_state=21
):
    """
    Run SCYAN with optional subsetting of adata.

    Parameters
    ----------
    adata : AnnData
        Full dataset.
    knowledgetable : str
        Filename of knowledge table (.xlsx).
    knowledgetable_dir : str
        Directory containing the knowledge table.
    base_dir : str
        Project base directory.
    batch_key : str
        obs column to use as batch key.
    unassigned_label : str
        Label to use for unassigned cells.
    phenotype_level : str, default="region"
        Name of column to store SCYAN predictions in.
    subset_key : str, optional
        obs column name to subset on (e.g., "region").
    subset_values : list, optional
        Values within `subset_key` to keep (e.g., ["Muscle"]).
    """

    random.seed(random_state)
    np.random.seed(seed=random_state)
    torch.manual_seed(seed=random_state)
    torch.cuda.manual_seed_all(seed=random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load knowledge table
    kt_path = os.path.join(base_dir, knowledgetable_dir, knowledgetable)
    kt = pd.read_excel(kt_path)
    kt_colnames = [col for col in kt.columns]
    pop_names = [col for col in kt_colnames if col not in adata.var_names.tolist()]
    kt.set_index(pop_names, inplace=True)
    assert kt.index.is_unique

    # --- Build combined mask ---
    mask = pd.Series(True, index=adata.obs.index)

    # categorical filter
    if filter is not None and filter_for is not None:
        if isinstance(filter_for, str):
            ff = [filter_for]
        else:
            ff = filter_for
        mask &= adata.obs[filter].isin(ff)

    # numeric Area filter
    if max_area is not None:
        mask &= adata.obs["Area"] < max_area

    # subset
    subset = adata[mask].copy()
    print(f"Subsetted AnnData to {subset.shape[0]} cells.")
    if filter_for is not None:
        print(f"Filtered for {filter_for} from {filter}")
    if max_area is not None:
        print(f"Filtered for maximal cell size of {max_area}")

    print(f"Running SCYAN on {subset.shape[0]} cells...")

    # # Prepare SCYAN input
    # X_input = subset.layers['counts'].copy()  # minimal copy for computation
    # if polarity:
    #     X_input *= subset.layers['polarity']

    # subset.X = np.arcsinh(X_input - 1)
    # subset.X = subset.X / subset.X.std(axis=0)

    X_input = subset.layers['corrected'].copy()
    if polarity:
        X_input *= subset.layers['polarity']
    subset.X = X_input / X_input.std(axis=0)

    del X_input
    gc.collect()
    
    # Initialize and fit model
    model = scyan.Scyan(subset, kt, batch_key=batch_key, continuous_covariates=continuous_covariates, prior_std=prior_std, lr=lr)
    model.fit(max_epochs=max_epochs, patience=patience, min_delta=min_delta)
    model.predict(key_added=phenotype_level, log_prob_th=log_prob_th, )

    print("Model run finished!")

    # Proportions/numbers for the subset run
    proportions = subset.obs[phenotype_level].value_counts(normalize=True, dropna=False)
    numbers = subset.obs[phenotype_level].value_counts(dropna=False)
    print(f"\nProportions by '{phenotype_level}' (subset):\n", proportions)
    print(f"\nAbsolute counts by '{phenotype_level}' (subset):\n", numbers)

    # Fill unassigned
    # Make sure unassigned_label is only added if not already a category
    if unassigned_label not in subset.obs[phenotype_level].cat.categories:
        subset.obs[phenotype_level] = subset.obs[phenotype_level].cat.add_categories([unassigned_label])
    subset.obs[phenotype_level] = subset.obs[phenotype_level].fillna(unassigned_label)

    if phenotype_level not in adata.obs:
        adata.obs[phenotype_level] = pd.Categorical(
            [np.nan] * adata.n_obs  # or fill with unassigned_label if you prefer
        )
    
    missing = [cat for cat in subset.obs[phenotype_level].cat.categories 
           if cat not in adata.obs[phenotype_level].cat.categories]
    if missing:
        adata.obs[phenotype_level] = adata.obs[phenotype_level].cat.add_categories(missing)

    # Integrate results back into original adata
    if filter is None and filter_for is None:
        adata.obs.loc[subset.obs_names, phenotype_level] = subset.obs[phenotype_level]
        adata.obs[phenotype_level] = adata.obs[phenotype_level].cat.remove_unused_categories()
    
    else:
        common_indices = subset.obs.index.intersection(adata.obs.index)
        new_categories = subset.obs.loc[common_indices, phenotype_level].astype(str).unique()

        # Add missing categories to adata
        adata.obs[phenotype_level] = adata.obs[phenotype_level].cat.add_categories(
            [cat for cat in new_categories if cat not in adata.obs[phenotype_level].cat.categories]
        )

        # Update predictions back into adata
        adata.obs.loc[common_indices, phenotype_level] = subset.obs.loc[common_indices, phenotype_level].astype(str)

        # Clean categories if needed
        adata.obs[phenotype_level] = adata.obs[phenotype_level].cat.remove_unused_categories()
    
    del subset
    gc.collect()
    return adata



# FlowSOM

# --- Helper: subset AnnData by markers + condition ---
def create_subadata(adata, markers, use, filter_for=None):
    channel_mask = adata.var.index.isin(markers)

    if filter_for is not None:
        if isinstance(filter_for, list):
            cell_mask = adata.obs[use].isin(filter_for)
        else:
            cell_mask = adata.obs[use] == filter_for
    else:
        cell_mask = adata.obs[use].notna()

    X = np.arcsinh(adata.layers['counts'][cell_mask][:, channel_mask])
    
    return ad.AnnData(
        X=X, 
        var=adata.var[channel_mask].copy(), 
        obs=adata.obs[cell_mask].copy()
    )


# --- Main: SOM + meta-clustering ---
def apply_som_and_meta_clustering(adata, n_clusters=30, som_grid_size=(10, 10)):
    start_time = time.time()

    # 1. Scale
    print("Scaling data...")
    X = adata.X
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    X_scaled = StandardScaler().fit_transform(X_scaled)

    # 2. SOM
    print(f"Training SOM grid={som_grid_size}...")
    som = MiniSom(som_grid_size[0], som_grid_size[1], X_scaled.shape[1],
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_batch(X_scaled, 10000)

    # Map cells to SOM nodes
    som_clusters = np.array([som.winner(x)[0] * som_grid_size[1] + som.winner(x)[1] for x in X_scaled])

    # 3. Meta-clustering (KMeans on SOM nodes)
    print(f"Meta-clustering into {n_clusters} clusters...")
    som_node_positions = np.array([som.winner(x) for x in X_scaled])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(som_node_positions)
    meta_labels = kmeans.labels_

    # Assign labels back to cells
    adata.obs['meta_labels'] = pd.Categorical(meta_labels.astype(str))

    print(f"Done. Total time: {time.time() - start_time:.2f} sec")
    return adata


# --- Plotting ---
def plot_expression_heatmap(adata, group_variable):
    df = pd.DataFrame(adata.X, columns=adata.var.index)
    df['Cluster'] = adata.obs[group_variable].values

    mean_exprs = df.groupby('Cluster').mean()
    counts = df['Cluster'].value_counts().reindex(mean_exprs.index)

    labels = [f"{c} (n={counts[c]})" for c in mean_exprs.index]

    plt.figure(figsize=(16, 8))
    sns.heatmap(mean_exprs, cmap="viridis", cbar=True, yticklabels=labels)
    plt.title(f"Mean marker expression per {group_variable}")
    plt.xlabel("Markers")
    plt.ylabel("Clusters")
    plt.tight_layout()
    plt.show()
