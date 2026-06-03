import os
import tifffile
from pathlib import Path
from typing import Sequence, Union
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import gc
import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def measure_weighted_centroids_all_channels(img: np.ndarray, mask: np.ndarray):
    """
    Compute weighted centroids for all channels at once.
    
    Returns:
        weighted_x: array of shape [num_objects, num_channels]
        weighted_y: array of shape [num_objects, num_channels]
        object_ids: array of object IDs
    """
    object_ids = np.unique(mask[mask != 0])
    num_objects = len(object_ids)
    num_channels = img.shape[0]

    weighted_x = np.zeros((num_objects, num_channels), dtype=float)
    weighted_y = np.zeros((num_objects, num_channels), dtype=float)

    for i in range(num_channels):
        props = regionprops(mask, intensity_image=img[i])
        weighted_x[:, i] = [p.weighted_centroid[0] for p in props]
        weighted_y[:, i] = [p.weighted_centroid[1] for p in props]

    return weighted_x, weighted_y, object_ids


def calculate_polarity_vectorized(regionprop_df, weighted_x, weighted_y, channel_names=None):
    """
    Compute polarity for all channels in a fully vectorized way.
    """
    centroids = regionprop_df[["centroid-0", "centroid-1"]].values
    major_axes = (regionprop_df["axis_major_length"].values / 2)[:, np.newaxis]
    minor_axes = (regionprop_df["axis_minor_length"].values / 2)[:, np.newaxis]
    orientations = np.radians(regionprop_df["eccentricity"].values)[:, np.newaxis]

    dx = weighted_x - centroids[:, 0:1]
    dy = weighted_y - centroids[:, 1:2]
    d_weighted = np.sqrt(dx**2 + dy**2)

    cos_theta = np.cos(orientations)
    sin_theta = np.sin(orientations)

    x_w_prime = cos_theta * dx + sin_theta * dy
    y_w_prime = -sin_theta * dx + cos_theta * dy

    t = 1 / np.sqrt((x_w_prime**2 / major_axes**2) + (y_w_prime**2 / minor_axes**2))
    d_perimeter = t * d_weighted

    centrality_percentage = np.clip(d_perimeter - d_weighted, 0, None) / d_perimeter
    centrality_percentage[np.isnan(centrality_percentage)] = 1.0

    # Use channel names if provided
    if channel_names is None:
        column_names = [f"Channel_{i}" for i in range(weighted_x.shape[1])]
    else:
        # Ensure the number of names matches number of channels
        if len(channel_names) != weighted_x.shape[1]:
            raise ValueError(
                f"Number of channel names ({len(channel_names)}) does not match number of channels in image ({weighted_x.shape[1]})."
            )
        column_names = channel_names

    polarity_df = pd.DataFrame(centrality_percentage, columns=column_names)
    polarity_df.insert(0, "Object", regionprop_df["Object"].values)

    return polarity_df


def process_image_mask(img_file, mask_file, regionprop_file, polarity_dir, channel_names=None):
    """
    Process one image + mask + regionprop CSV and save polarity.
    """
    img = tifffile.imread(img_file)  # [channels, H, W]
    mask = tifffile.imread(mask_file)  # [H, W]
    regionprop_df = pd.read_csv(regionprop_file)

    # Compute weighted centroids
    weighted_x, weighted_y, object_ids = measure_weighted_centroids_all_channels(img, mask)

    # Compute polarity using panel channel names
    polarity_df = calculate_polarity_vectorized(
        regionprop_df, weighted_x, weighted_y, channel_names=channel_names
    )

    output_file = Path(polarity_dir) / f"{Path(img_file).stem}.csv"
    polarity_df.to_csv(output_file, index=False)

    del img, mask, regionprop_df, weighted_x, weighted_y, polarity_df
    gc.collect()


def create_polarity_files(base_dir: str, img_dir: str = "img_comp", mask_dir: str = "masks_merged", overwrite: bool = False):
    """
    Run the full polarity workflow for all images.
    """
    # Load panel
    panel = pd.read_csv(os.path.join(base_dir, "panel.csv"))
    channel_names = list(panel['name'])

    img_dir = Path(base_dir) / img_dir
    mask_dir = Path(base_dir) / mask_dir
    regionprop_dir = Path(base_dir) / "regionprops"
    polarity_dir = Path(base_dir) / "polarity"
    polarity_dir.mkdir(exist_ok=True)

    img_files = sorted(img_dir.glob("*.tiff"))
    mask_files = sorted(mask_dir.glob("*.tiff"))
    regionprop_files = sorted(regionprop_dir.glob("*.csv"))

    file_triplets = list(zip(img_files, mask_files, regionprop_files))
    random.shuffle(file_triplets)

    for img_file, mask_file, regionprop_file in file_triplets:
        output_file = polarity_dir / f"{img_file.stem}.csv"
        
        if not overwrite and output_file.exists():
            print(f"Skipping {img_file.name}: file already exists.")
            continue
        try:
            # Process image with actual channel names
            process_image_mask(img_file, mask_file, regionprop_file, polarity_dir, channel_names=channel_names)
            print(f"Processed {img_file.name}")
        except Exception as e:
            logger.exception(f"Error processing {img_file}: {e}")
