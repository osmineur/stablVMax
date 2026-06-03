import os
import tifffile
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.morphology import remove_small_holes, remove_small_objects, dilation, erosion, disk
from scipy.ndimage import convolve, gaussian_filter
from skimage.measure import regionprops
from typing import Dict
from collections import defaultdict


def kernel_binary_filter(image, threshold=1.0, neighbor_threshold=2):
    """Detect tissue using local neighborhood filtering."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    low_center_pixels = image < threshold
    high_neighbors = (image > threshold).astype(int)
    neighbor_count = convolve(high_neighbors, kernel, mode='constant', cval=0)

    binary_mask = np.ones_like(image, dtype=bool)
    binary_mask[low_center_pixels & (neighbor_count < neighbor_threshold)] = 0
    return binary_mask

def generate_binary_mask(
    img_channel,
    arcsinh_transform=True,
    filter_threshold=1.0,
    neighbor_threshold=2,
    dilation_radius=2,
    erosion_radius=2,
    hole_area=500,
    min_size=1000,
    smooth_sigma=5
):
    """Generate a cleaned binary mask from a single image channel."""
    
    # Step 1: Transform signal
    if arcsinh_transform:
        signal = np.arcsinh(img_channel.astype(np.float32))
    else:
        signal = img_channel.astype(np.float32)
    
    # Step 2: Initial binary filter
    mask = kernel_binary_filter(signal, threshold=filter_threshold, neighbor_threshold=neighbor_threshold)
    
    # Step 3: Morphology + cleanup
    if dilation_radius > 0:
        mask = dilation(mask, disk(dilation_radius))
    if erosion_radius > 0:
        mask = erosion(mask, disk(erosion_radius))
    
    mask = remove_small_holes(mask, area_threshold=hole_area)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = gaussian_filter(mask.astype(float), sigma=smooth_sigma) > 0.5

    return mask

def create_compartment_mask(panck_mask, tissue_mask, selem_radius=20):
    """Create a labeled compartment mask from tumor and tissue masks."""
    selem = disk(selem_radius)
    
    tumor_core = erosion(panck_mask, selem)
    tumor_border = panck_mask & ~tumor_core
    stroma_core = ~dilation(panck_mask, selem)
    stroma_border = tissue_mask & ~panck_mask & ~stroma_core

    # Initialize mask
    final_mask = np.zeros_like(panck_mask, dtype=np.uint8)
    
    # Apply labels
    zones = [
        (tissue_mask & stroma_core,    1),
        (tissue_mask & stroma_border,  2),
        (tissue_mask & tumor_border,   3),
        (tissue_mask & tumor_core,     4),
    ]
    
    for zone_mask, label in zones:
        final_mask[zone_mask] = label

    return final_mask



def generate_compartment_mask(img_path: str, save_dir: str, show=False):
    
    img = tifffile.imread(img_path)

    # --- Step 1: Tissue detection (channel 2) ---
    tissue_mask = generate_binary_mask(
        img_channel=img[2],
        filter_threshold=0.5,
        neighbor_threshold=4,
        dilation_radius=2,
        erosion_radius=2,
        hole_area=500,
        min_size=1000,
        smooth_sigma=5
    )

    if show:
        plt.imshow(tissue_mask, cmap='gray')
        plt.title("Tissue Mask")
        plt.show()


    # --- Step 2: Tumor detection (PanCK, channel 0) ---
    panck_mask = generate_binary_mask(
        img_channel=img[0],
        filter_threshold=2,
        neighbor_threshold=2,
        dilation_radius=2,
        erosion_radius=2,
        hole_area=500,
        min_size=200,
        smooth_sigma=5
    )
    if show:
        plt.imshow(panck_mask, cmap='gray')
        plt.title("Tumor Mask")
        plt.show()

    final_mask = create_compartment_mask(panck_mask, tissue_mask, selem_radius=20)

    # Filter out small tissue islands
    final_mask[~remove_small_objects(final_mask > 0, min_size=1000)] = 0

    if show:
        plt.imshow(final_mask, cmap='nipy_spectral')
        plt.title("Final Compartment Mask")
        plt.colorbar()
        plt.show()

    # Save
    out_path = os.path.join(save_dir, os.path.basename(img_path))
    tifffile.imwrite(out_path, final_mask)



def batch_generate_compartment_masks(base_dir, overwrite: bool = False, show: bool = False):
    img_dir = os.path.join(base_dir, "img_comp")
    save_dir = os.path.join(base_dir, "masks_compartments")
    os.makedirs(save_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff') if'T59' in str(f)]
    random.shuffle(img_files)


    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(save_dir, img_file)

        if not overwrite and os.path.exists(mask_path):
            print(f"Skipping existing file: {img_file}")
            continue

        print(f"Processing {img_file}...")
        generate_compartment_mask(img_path, save_dir, show=show)


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

        # Count pixels for each zone label (1 to 4)
        stromacore = np.sum(compartment_mask == 1)
        stromaborder = np.sum(compartment_mask == 2)
        tumorborder = np.sum(compartment_mask == 3)
        tumorcore = np.sum(compartment_mask == 4)

        # Append the row to the DataFrame
        table.append({
            "Image": image_name,
            "Stromacore": stromacore,
            "Stromaborder": stromaborder,
            "Tumorborder": tumorborder,
            "Tumorcore": tumorcore
        })
    zonal_areas_table = pd.DataFrame(table)
    zonal_areas_table.to_csv(os.path.join(base_dir, "zonal_areas_summary.csv"), index=False)
    return zonal_areas_table