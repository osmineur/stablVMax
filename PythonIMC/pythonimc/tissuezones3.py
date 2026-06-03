import os
import random
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import erosion, dilation, disk, remove_small_objects, remove_small_holes, binary_erosion
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.measure import regionprops_table
from tqdm import tqdm




def get_channels_sum(img, channel_index):
    if isinstance(channel_index, int):
        return img[channel_index]

    # If it's a list, sum the selected channels
    return img[channel_index].max(axis=0)

def channel_mask(
    img,
    channel_index,
    sigma_blur: int = 4,
    weight_sigma1: float = 2.0,
    threshold: float = 0.2,
    fallback_threshold: float = 0.5,
    erosion_value: int = None,
    min_object_size: int = None,
    fill_hole_size: int = None,
    plot: bool = False
):
    """Compute a binary mask for a single channel of an image."""
    ch = get_channels_sum(img, channel_index)
    blur = gaussian_filter(ch, sigma=sigma_blur)          
    blur = np.arcsinh(blur)

    # Fit 2-component GMM
    values = blur.ravel().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(values)
    means = gmm.means_.flatten()
    order = np.argsort(means)
    mu1, mu2 = means[order]

    # Weighted midpoint threshold
    thresh_val = (mu1 * weight_sigma1 + mu2) / (weight_sigma1 + 1)
    if abs(mu2 - mu1) < threshold:
        thresh_val = fallback_threshold
    
    if plot:
        plt.hist(values, bins=200, density=True, alpha=0.5, color="gray")
        plt.axvline(mu1, color="red", linestyle="--", label=f"μ1={mu1:.2f}")
        plt.axvline(mu2, color="blue", linestyle="--", label=f"μ2={mu2:.2f}")
        plt.axvline(thresh_val, color="k", linestyle="-", label="Threshold")
        plt.ylim(0, 2)
        plt.legend()
        plt.title(f"Histogram of channel index {channel_index}")
        plt.show()

    mask = blur > thresh_val

    if erosion_value:
        mask = erosion(mask, disk(erosion_value))
    if min_object_size:
        mask = remove_small_objects(mask, min_size=min_object_size)
    if fill_hole_size:
        mask = remove_small_holes(mask, area_threshold=fill_hole_size)

    return mask


def generate_channel_masks(
    img,
    tissue_channel: int = None,
    tumor_channel: int = None,
    muscle_channel: int = None,
    plot: bool = False
):
    """Generate masks for tissue, tumor, and muscle channels."""
    tissuemask = tumormask = musclemask = None

    if tissue_channel is not None:
        tissuemask = channel_mask(img, channel_index=tissue_channel, erosion_value=4,
                                  min_object_size=5000, fill_hole_size=500, plot=plot)
    if tumor_channel is not None:
        tumormask = channel_mask(img, channel_index=tumor_channel, min_object_size=250, plot=plot)
    if muscle_channel is not None:
        musclemask = channel_mask(img, channel_index=muscle_channel, weight_sigma1=0.5,
                                  min_object_size=250, threshold = 0.8, fallback_threshold=1.5, plot=plot)
    
    return tissuemask, tumormask, musclemask


def generate_compartment_masks(
    img,
    tissue_channel: int = None,
    tumor_channel: int = None,
    muscle_channel: int = None,
    plot: bool = False,
    selem_radius: int = 20
):
    """Generate compartment masks from a single image array."""
    tissue_mask, tumor_mask, muscle_mask = generate_channel_masks(
        img,
        tissue_channel=tissue_channel,
        tumor_channel=tumor_channel,
        muscle_channel=muscle_channel,
        plot=plot
    )
    
    selem = disk(selem_radius)

    # Tumor subregions
    tumor_core = tissue_mask & erosion(tumor_mask, selem)
    tumor_border = tissue_mask & tumor_mask & ~tumor_core
    stroma_core = ~dilation(tumor_mask, selem) & tissue_mask
    stroma_border = tissue_mask & ~tumor_mask & ~stroma_core

    # Muscle zones
    muscle_core = stroma_core & muscle_mask
    muscle_border = stroma_border & muscle_mask
    muscle_tumor_border = tumor_border & muscle_mask
    muscle_tumor_core = tumor_core & muscle_mask

    # Remove muscle from global zones
    stroma_core &= ~muscle_mask
    stroma_border &= ~muscle_mask
    tumor_core &= ~muscle_mask
    tumor_border &= ~muscle_mask

    # Build final mask
    final_mask = np.zeros_like(tissue_mask, dtype=np.uint16)
    zones = [
        (stroma_core,        11),
        (stroma_border,      12),
        (muscle_core,        21),
        (muscle_border,      22),
        (muscle_tumor_border,23),
        (muscle_tumor_core,  24),
        (tumor_border,       33),
        (tumor_core,         34),
    ]
    for zone_mask, label in zones:
        final_mask[zone_mask] = label

    if plot:
        labels = [0, 11, 12, 21, 22, 23, 24, 33, 34]
        colors = ["black", "lightblue", "blue", "lightgreen", "green",
                  "orange", "red", "purple", "magenta"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(labels + [max(labels)+1], cmap.N)
        plt.figure(figsize=(10,10))
        im = plt.imshow(final_mask, cmap=cmap, norm=norm)
        plt.axis("off")
        plt.colorbar(im, ticks=labels)
        plt.show()

    return final_mask


def batch_generate_compartment_masks(
    base_dir,
    tissue_channel: int = None,
    tumor_channel: int = None,
    muscle_channel: int = None,
    overwrite: bool = False,
    plot: bool = False,
    img_folder: str = "img_comp",
    mask_folder: str = "masks_compartments",
    img_name: str = None
):
    """Loop over a directory, read TIFFs, run compartment generation, and save."""
    img_dir = os.path.join(base_dir, img_folder)
    save_dir = os.path.join(base_dir, mask_folder)
    os.makedirs(save_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff')]
    random.shuffle(img_files)

    if img_name:
        img_files = [f for f in img_files if img_name in str(f)]

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(save_dir, img_file)

        if not overwrite and os.path.exists(mask_path):
            print(f"Skipping existing file: {img_file}")
            continue

        print(f"Processing {img_file}...")
        img_array = tifffile.imread(img_path)

        final_mask = generate_compartment_masks(
            img=img_array,
            tissue_channel=tissue_channel,
            tumor_channel=tumor_channel,
            muscle_channel=muscle_channel,
            plot=plot,
            selem_radius=20
        )

        tifffile.imsave(mask_path, final_mask.astype(np.uint16))



def zonal_areas_table(base_dir):
    compartment_masks_dir = os.path.join(base_dir, "masks_compartments")
    compartment_mask_files = sorted([f for f in os.listdir(compartment_masks_dir) if f.endswith(".tiff")])

    table = []

    for f in compartment_mask_files:
        image_id = os.path.splitext(f)[0]
        image_name = os.path.splitext(image_id)[0].split('_')[-1]

        compartment_mask_path = os.path.join(compartment_masks_dir, f)
        
        if not os.path.exists(compartment_mask_path):
            print(f"Zone mask not found for {f}, skipping.")
            continue

        compartment_mask = tifffile.imread(compartment_mask_path)
        #compartment_mask = compartment_mask % 10

        # Count pixels for each zone label (1 to 4)
        stromacore = np.sum(compartment_mask == 11)
        stromaborder = np.sum(compartment_mask == 12)
        tumorborder = np.sum(compartment_mask == 23) + np.sum(compartment_mask == 33)
        tumorcore = np.sum(compartment_mask == 24) + np.sum(compartment_mask == 34)
        musclearea = np.sum(compartment_mask == 21) + np.sum(compartment_mask == 22)

        # Append the row to the DataFrame
        table.append({
            "Image": image_name,
            "Stromacore": stromacore,
            "Stromaborder": stromaborder,
            "Tumorborder": tumorborder,
            "Tumorcore": tumorcore,
            "Stromamuscle": musclearea
        })
    zonal_areas_table = pd.DataFrame(table)
    zonal_areas_table.to_csv(os.path.join(base_dir, "zonal_areas_summary.csv"), index=False)
    return zonal_areas_table




def calculate_distance_to_tumor(base_dir,
                                adata,
                                columnname: str = 'Distance_to_tumor',
                                masks_folder: str = "masks_merged",
                                masks_compartments_folder: str = "masks_compartments",
                                overwrite: bool = False,
                                ):
        
    # choose your structuring element for erosion
    selem = disk(1)

    # create a new column for distance
    if columnname not in adata.obs.columns:
        adata.obs[columnname] = np.nan
    
    if overwrite:
        images_to_process = adata.obs['Image'].unique()
    else:
        # only images with NaNs in the column
        images_to_process = adata.obs.loc[adata.obs[columnname].isna(), 'Image'].unique()


    # get list of files
    comp_dir = os.path.join(base_dir, masks_compartments_folder)
    if not os.path.isdir(comp_dir):
        print(f"Skipping calculate_distance_to_tumor: '{masks_compartments_folder}/' not found.")
        print("Run the tissue zones step first to generate compartment masks.")
        return adata

    mask_files = os.listdir(os.path.join(base_dir, masks_folder))
    compartment_files = os.listdir(comp_dir)

    for img_id in tqdm(images_to_process, desc="Calculating distance to tumor"):
        
        # find the corresponding mask file
        mask_file = [f for f in mask_files if img_id in f][0]
        compartment_file = [f for f in compartment_files if img_id in f][0]
        
        # load masks
        mask = tifffile.imread(os.path.join(base_dir, "masks_merged", mask_file))
        compartmentmask = tifffile.imread(os.path.join(base_dir, "masks_compartments", compartment_file))
        
        # binarize tumor
        last_digit = compartmentmask % 10
        tumor_mask = np.isin(last_digit, [3,4]).astype(np.uint8)
        
        # border
        eroded = binary_erosion(tumor_mask, selem)
        border = tumor_mask ^ eroded
        
        # distance transform
        dist_px = distance_transform_edt(border == 0)
        
        # signed distance: positive inside tumor, negative outside
        dist_signed = dist_px.copy()
        dist_signed[~tumor_mask.astype(bool)] *= -1
        
        # regionprops: per-cell mean distance
        props = regionprops_table(
            mask.astype(int),
            intensity_image=dist_signed,
            properties=['label', 'mean_intensity']
        )
        df = pd.DataFrame(props)
        df['UniqueCellID'] = img_id + "_" + df['label'].astype(str)
        df.rename(columns={'mean_intensity':'mean_dist_px'}, inplace=True)
        
        idx = adata.obs['Image'] == img_id
        obs_img = adata.obs[idx].copy()
        
        obs_img = obs_img.merge(df[['UniqueCellID','mean_dist_px']], on='UniqueCellID', how='left')
        
        adata.obs.loc[idx, columnname] = obs_img['mean_dist_px'].values

    return adata
