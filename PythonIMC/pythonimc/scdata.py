import os
import gc
import random
import numpy as np
import pandas as pd
import tifffile
import logging
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Sequence, Union, Generator, Tuple
import scipy.ndimage
from skimage.measure import regionprops_table
from skimage.measure import regionprops
from collections import defaultdict


logger = logging.getLogger(__name__)

class IntensityAggregation(Enum):
    SUM = partial(scipy.ndimage.sum_labels)
    MIN = partial(scipy.ndimage.minimum)
    MAX = partial(scipy.ndimage.maximum)
    MEAN = partial(scipy.ndimage.mean)
    MEDIAN = partial(scipy.ndimage.median)
    STD = partial(scipy.ndimage.standard_deviation)
    VAR = partial(scipy.ndimage.variance)

def measure_intensities(
    img: np.ndarray,
    mask: np.ndarray,
    channel_names: Sequence[str],
    intensity_aggregation: IntensityAggregation,
) -> pd.DataFrame:
    object_ids = np.unique(mask[mask != 0])
    data = {
        channel_name: intensity_aggregation.value(img[i], labels=mask, index=object_ids)
        for i, channel_name in enumerate(channel_names)
    }
    return pd.DataFrame(data, index=pd.Index(object_ids, name="Object"))



def try_measure_intensities_from_disk(
    img_files: Sequence[Union[str, Path]],
    mask_files: Sequence[Union[str, Path]],
    channel_names: Sequence[str],
    intensity_aggregation: IntensityAggregation,
) -> Generator[Tuple[Path, Path, pd.DataFrame], None, None]:
    for img_file, mask_file in zip(img_files, mask_files):
        try:
            img = tifffile.imread(img_file)
            mask = tifffile.imread(mask_file)

            intensities = measure_intensities(img, mask, channel_names, intensity_aggregation)
            yield Path(img_file), Path(mask_file), intensities

            # Clean up
            del img, mask, intensities
        except Exception as e:
            logging.exception(f"Error measuring intensities in {img_file}: {e}")


def create_intensities_files(
    base_dir,
    image_folder='img_comp',
    masks_folder='masks_merged',
    aggregation='mean'
):
    # Read panel
    panel = pd.read_csv(os.path.join(base_dir, "panel.csv"))
    channel_names = list(panel['name'])

    # Aggregation method
    aggregation = aggregation.lower()
    if aggregation not in ("mean", "sum"):
        raise ValueError("Aggregation must be 'mean' or 'sum'.")

    intensity_aggregation = (
        IntensityAggregation.MEAN if aggregation == 'mean' else IntensityAggregation.SUM
    )

    # Set directories
    img_dir = os.path.join(base_dir, image_folder)
    masks_dir = os.path.join(base_dir, masks_folder)
    output_dir = os.path.join(base_dir, "intensities")
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted files
    image_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".tiff"))
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tiff"))

    # Build lookup by base name
    mask_dict = {os.path.splitext(f)[0]: f for f in mask_files}

    # Pair only existing masks
    file_pairs = []
    for img in image_files:
        base = os.path.splitext(img)[0]
        if base in mask_dict:  # only add if matching mask exists
            file_pairs.append((img, mask_dict[base]))

    # Shuffle
    random.shuffle(file_pairs)

    print(f"Found {len(file_pairs)} image–mask pairs")

    for img_file, mask_file in file_pairs:
        if img_file != mask_file:
            raise ValueError(f"Mask and image don't match: {img_file} vs {mask_file}")

        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        output_filename = os.path.splitext(img_file)[0] + ".csv"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Output {output_filename} exists. Skipping.")
            continue

        for _, _, intensities_df in try_measure_intensities_from_disk(
            img_files=[img_path],
            mask_files=[mask_path],
            channel_names=channel_names,
            intensity_aggregation=intensity_aggregation
        ):
            intensities_df.to_csv(output_path)
            print(f"Saved intensities: {output_filename}")
            gc.collect()






def measure_regionprops(
    img: np.ndarray,
    mask: np.ndarray,
    skimage_regionprops: Sequence[str],
) -> pd.DataFrame:
    props = list(skimage_regionprops)
    if "label" not in props:
        props.insert(0, "label")

    data = regionprops_table(
        mask,
        intensity_image=np.moveaxis(img, 0, -1),  # Channel-last
        properties=props
    )
    object_ids = data.pop("label")
    return pd.DataFrame(data, index=pd.Index(object_ids, name="Object"))


def try_measure_regionprops(
    img_files: Sequence[Union[str, Path]],
    mask_files: Sequence[Union[str, Path]],
    skimage_regionprops: Sequence[str],
) -> Generator[Tuple[Path, Path, pd.DataFrame], None, None]:
    for img_file, mask_file in zip(img_files, mask_files):
        try:
            img = tifffile.imread(img_file)
            mask = tifffile.imread(mask_file)

            props = measure_regionprops(img, mask, skimage_regionprops)
            yield Path(img_file), Path(mask_file), props

            del img, mask, props
        except Exception as e:
            logger.exception(f"Error measuring regionprops in {img_file}: {e}")


def create_regionprops_files(
    base_dir,
    image_folder='img_comp',
    masks_folder='masks_merged',
    regionprops_list=None,
    compartment_masks_folder=None,
    zone_labels=None,
):
    if regionprops_list is None:
        regionprops_list = [
            "area", "centroid", "axis_major_length",
            "axis_minor_length", "eccentricity", "orientation"
        ]

    if zone_labels is None:
        zone_labels = {
            0: "Background",
            11: "Stroma_Stromacore",
            12: "Stroma_Stromaborder",
            21: "Muscle_Stromacore",
            22: "Muscle_Stromaborder",
            23: "Muscle_Tumorborder",
            24: "Muscle_Tumorcore",
            33: "Tumor_Tumorborder",
            34: "Tumor_Tumorcore"
        }

    # Set directories
    img_dir = os.path.join(base_dir, image_folder)
    masks_dir = os.path.join(base_dir, masks_folder)
    regionprops_dir = os.path.join(base_dir, "regionprops")
    os.makedirs(regionprops_dir, exist_ok=True)

    if compartment_masks_folder:
        compartment_dir = os.path.join(base_dir, compartment_masks_folder)
    else:
        compartment_dir = None

    # Load and pair files
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tiff')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.tiff')])
    file_pairs = list(zip(image_files, mask_files))
    random.shuffle(file_pairs)

    for img_file, mask_file in file_pairs:
        if img_file != mask_file:
            raise ValueError(f"Image and mask mismatch: {img_file} vs {mask_file}")

        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        output_filename = os.path.splitext(img_file)[0] + ".csv"
        output_path = os.path.join(regionprops_dir, output_filename)

        if os.path.exists(output_path):
            print(f"Output {output_filename} exists. Skipping.")
            continue

        # Load optional compartment mask
        if compartment_dir:
            compartment_path = os.path.join(compartment_dir, mask_file)
            if os.path.exists(compartment_path):
                zone_mask = tifffile.imread(compartment_path)
            else:
                print(f"Missing compartment mask for {mask_file}, skipping zone assignment.")
                zone_mask = None
        else:
            zone_mask = None

        # Process image/mask
        for _, _, regionprops_df in try_measure_regionprops(
            img_files=[img_path],
            mask_files=[mask_path],
            skimage_regionprops=regionprops_list
        ):
            if zone_mask is not None:
                mask = tifffile.imread(mask_path)
                cell_regions = regionprops(mask)
                zone_assignments = {}

                for cell in cell_regions:
                    coords = cell.coords
                    zones = zone_mask[coords[:, 0], coords[:, 1]]
                    dominant_zone = np.bincount(zones).argmax()
                    zone_name = zone_labels.get(dominant_zone, "unknown")
                    zone_assignments[cell.label] = zone_name

                # Add 'Zone' column to regionprops dataframe
                regionprops_df["Zone"] = regionprops_df.index.map(zone_assignments)

            regionprops_df.to_csv(output_path)
            print(f"Saved regionprops: {output_filename}")
            gc.collect()

