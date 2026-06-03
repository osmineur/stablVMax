import os
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from tqdm import tqdm


def assemble_anndata(base_dir, regionprops=True, add_polarity=False):
    # === Step 1: Load intensity data ===
    intensities_folder = os.path.join(base_dir, "intensities")
    intensity_files = [f for f in os.listdir(intensities_folder) if f.endswith(".csv")]

    all_data = []
    all_obs = []

    print("Loading intensity data...")
    for file in tqdm(intensity_files, desc="Reading intensities"):
        file_path = os.path.join(intensities_folder, file)
        df = pd.read_csv(file_path)
        
        image_id = os.path.splitext(file)[0].split('_')[-1]
        df["Image"] = image_id
        df["UniqueCellID"] = image_id + "_" + df["Object"].astype(str)  # Faster than apply()

        markers = df.drop(columns=["Object", "UniqueCellID", "Image"])
        obs = df[["UniqueCellID", "Object", "Image"]]

        all_data.append(markers)
        all_obs.append(obs)

    # Combine intensity data
    combined_data = pd.concat(all_data, axis=0, ignore_index=True).astype(np.float32)
    combined_obs = pd.concat(all_obs, axis=0, ignore_index=True)

    # Create AnnData object
    adata = ad.AnnData(X=combined_data.values)
    adata.obs = combined_obs
    adata.obs.set_index("UniqueCellID", inplace=True)
    adata.layers["counts"] = combined_data.values
    del adata.X

    # Build adata.var
    # Column names can be either 'channel_marker' (e.g. 'Ir191_DNA1') or
    # just 'marker' (e.g. 'DNA1') depending on how scdata generated them.
    adata.var['channel_marker'] = combined_data.columns
    adata.var['marker'] = [col.split('_')[1] if '_' in col else col for col in combined_data.columns]
    adata.var['channel'] = [col.split('_')[0] if '_' in col else '' for col in combined_data.columns]
    adata.var.index = adata.var['marker']

    print("AnnData object created successfully!")
    print(adata.shape)

    # Clean up
    del combined_data
    del all_data
    del all_obs

    if regionprops:
        # === Step 2: Load regionprops data ===
        regionprops_folder = os.path.join(base_dir, "regionprops")
        regionprops_files = [f for f in os.listdir(regionprops_folder) if f.endswith(".csv")]
        regionprops_data = []

        print("Loading regionprops data...")
        for file in tqdm(regionprops_files, desc="Reading regionprops"):
            file_path = os.path.join(regionprops_folder, file)
            region_df = pd.read_csv(file_path)

            image_id = os.path.splitext(file)[0].split('_')[-1]
            region_df["Image"] = image_id
            region_df["UniqueCellID"] = image_id + "_" + region_df["Object"].astype(str)

            regionprops_data.append(region_df)

        regionprops_df = pd.concat(regionprops_data, axis=0, ignore_index=True)
        regionprops_df = regionprops_df.drop(columns=["Object", "Image"]).set_index("UniqueCellID")

        # Merge regionprops with adata.obs
        adata.obs = adata.obs.merge(regionprops_df, how="left", left_index=True, right_index=True)
        adata.obs.columns = [col[0].upper() + col[1:] if col else col for col in adata.obs.columns]

        print("Regionprops data merged successfully!")

        # Clean up
        del regionprops_df
        del regionprops_data


    # === Step 3: Load polarity data (optional) ===
    if add_polarity:
        polarity_dir = os.path.join(base_dir, "polarity")
        if os.path.exists(polarity_dir):
            print("Loading polarity data...")
            all_polarity = []
            for file in tqdm(os.listdir(polarity_dir), desc="Reading polarity"):
                if not file.endswith(".csv"):
                    continue
                file_path = os.path.join(polarity_dir, file)
                df = pd.read_csv(file_path)
                image_id = os.path.splitext(file)[0].split('_')[-1]
                df["UniqueCellID"] = image_id + "_" + df["Object"].astype(str)
                df = df.set_index("UniqueCellID")
                # drop Object column
                df = df.drop(columns=["Object"], errors='ignore')
                all_polarity.append(df)

            if all_polarity:
                combined_polarity = pd.concat(all_polarity, axis=0, ignore_index=False)
                # align to adata.obs
                combined_polarity = combined_polarity.reindex(adata.obs.index)
                adata.layers['polarity'] = combined_polarity.values.astype(np.float32)
                print("Polarity layer added to adata.layers['polarity']")
                del all_polarity, combined_polarity
        else:
            print(f"No polarity folder found at {polarity_dir}, skipping polarity layer.")

    return adata


def anndata_qc(adata, output_dir="."):
    # 1. Check for 'Zone' column
    if "Zone" in adata.obs.columns:
        if "Background" in adata.obs["Zone"].values:
            n_bg = (adata.obs["Zone"] == "Background").sum()
            total_cells = adata.obs.shape[0]
            print(f"Cells assigned to background: {n_bg}. Total cell count: {total_cells}.")
        else:
            print("No cells assigned to zone 'Background' found.")
    else:
        print("Column 'Zone' not found in Anndata.")

    # 2. Check for 'Area' column
    if "Area" in adata.obs.columns:
        area = adata.obs["Area"]
        print(f"Minimum cell area: {area.min()}")
        print(f"Maximum cell area: {area.max()}")

        top10 = area.sort_values(ascending=False).head(10)
        print("\nTop 10 largest cells by area:")
        for uid, a in top10.items():
            print(f"  {uid}: {a:.2f}")

        # Plot histogram of area
        plt.figure(figsize=(6, 4))
        area_99 = np.percentile(area, 99)
        bins = np.linspace(0, area_99, 200)
        plt.hist(area, bins=bins, edgecolor='black')
        plt.xlabel('Cell Area')
        plt.ylabel('Number of Cells')
        plt.title('Distribution of Cell Areas')
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'area' not found in Anndata.")

    # 3. Marker sum from adata.layers['comp_exprs']

    marker_sum = np.sum(np.arcsinh(adata.layers['counts']), axis=1)

    # Get top 10 highest marker_sum cells
    top10_marker_sum = marker_sum.argsort()[-10:][::-1]  # indices of top 10 in descending order

    print("\nTop 10 cells with highest marker_sum:")
    for idx in top10_marker_sum:
        cell_id = adata.obs.index[idx]
        value = marker_sum[idx]
        print(f"  {cell_id}: {value:.2f}")
    
    # Plot histogram
    marker_99 = np.percentile(marker_sum, 99.9)
    bins = np.linspace(0, marker_99, 100)

    plt.figure(figsize=(6, 4))
    plt.hist(marker_sum, bins=bins, edgecolor='black')
    plt.xlabel('Marker Sum')
    plt.ylabel('Number of Cells')
    plt.tight_layout()
    plt.show()


def add_metadata(
    adata,
    metadata_files,
    base_dir: str,
    metadata_dir: str = "metadata",
    merge_on_column: str = 'Image'
):
    """
    Merge patient metadata into `adata.obs` based on a specified column.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to update.
    metadata_files : list of str
        List of CSV filenames containing metadata.
    metadata_dir : str
        Subdirectory (within base_dir) where metadata files are located.
    base_dir : str
        Base directory path.
    merge_on : str
        Column name used to merge the metadata into adata.obs.

    Returns
    -------
    adata : anndata.AnnData
        The updated AnnData object with metadata merged into `.obs`.
    """
    
    # Construct full metadata folder path
    metadata_folder = os.path.join(base_dir, metadata_dir)

    metadata = []
    for file in metadata_files:
        file_path = os.path.join(metadata_folder, file)
        if file.lower().endswith((".csv")):
            df = pd.read_csv(file_path)
        elif file.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file}")
        metadata.append(df)

    metadata = pd.concat(metadata, axis=0, ignore_index=True)
        
    if merge_on_column not in metadata.columns:
        raise ValueError(f"Column '{merge_on_column}' not found in metadata.")

    if merge_on_column not in adata.obs.columns:
        raise ValueError(f"Column '{merge_on_column}' not found in adata.obs.")

    # Merge into adata.obs
    adata.obs[merge_on_column] = adata.obs[merge_on_column].astype(str)
    metadata[merge_on_column] = metadata[merge_on_column].astype(str)

    # --- Prevent duplicate columns ---
    overlapping_cols = set(metadata.columns) & set(adata.obs.columns)
    overlapping_cols.discard(merge_on_column)  # keep the merge key
    if overlapping_cols:
        print(f"Dropping overlapping columns from metadata: {list(overlapping_cols)}")
        metadata = metadata.drop(columns=overlapping_cols)

    adata.obs = adata.obs.merge(
        metadata.set_index(merge_on_column),
        how='left',
        left_on=merge_on_column,
        right_index=True
    )
    adata.obs.columns = [col[0].upper() + col[1:] if col else col for col in adata.obs.columns]
    
    for col in adata.obs.select_dtypes(include=['object']).columns:
        adata.obs[col] = adata.obs[col].astype(str)

    return adata

