import os
import random
import gc
import tifffile
import logging

import pandas as pd
import numpy as np

from enum import Enum
from functools import partial
from pathlib import Path
from typing import Generator, Optional, Sequence, Tuple, Union
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist, squareform
from skimage.measure import regionprops


logger = logging.getLogger(__name__)

def _expand_mask_euclidean(mask: np.ndarray, dmax: float) -> np.ndarray:
    dists, (i, j) = distance_transform_edt(mask == 0, return_indices=True)
    return np.where(dists <= dmax, mask[i, j], mask)


def _to_triu_indices(k: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # https://stackoverflow.com/a/27088560 (optimized for numpy arrays)
    # Briefly, the formulas were derived using triangular numbers/roots
    i = n - 2 - np.floor((-8 * k + 4 * n * (n - 1) - 7) ** 0.5 / 2 - 0.5)
    j = k + i + 1 - (n * (n - 1) - (n - i) * (n - i - 1)) // 2
    return i.astype(int), j.astype(int)


def _measure_centroid_distance_neighbors(
    mask: np.ndarray,
    metric: Optional[str] = None,
    dmax: Optional[float] = None,
    kmax: Optional[int] = None,
) -> pd.DataFrame:
    if metric is None:
        raise ValueError("Metric is required")
    props = regionprops(mask)
    labels = np.array([p.label for p in props])
    centroids = np.array([p.centroid for p in props])
    condensed_dists = pdist(centroids, metric=metric)
    if kmax is not None:
        k = min(kmax, len(props) - 1)
        dist_mat = squareform(condensed_dists, checks=False).astype(float)
        np.fill_diagonal(dist_mat, np.inf)
        knn_indices = np.argpartition(dist_mat, k - 1)[:, :k]
        if dmax is not None:
            knn_dists = np.take_along_axis(dist_mat, knn_indices, -1)
            indices1, indices2 = np.nonzero(knn_dists <= dmax)
            indices2 = knn_indices[(indices1, indices2)]
        else:
            indices1 = np.repeat(np.arange(len(props)), k)
            indices2 = np.ravel(knn_indices)
        distances = dist_mat[(indices1, indices2)]
    elif dmax is not None:
        (condensed_indices,) = np.nonzero(condensed_dists <= dmax)
        indices1, indices2 = _to_triu_indices(condensed_indices, len(props))
        distances = condensed_dists[condensed_indices]
        indices1, indices2, distances = (
            np.concatenate((indices1, indices2)),
            np.concatenate((indices2, indices1)),
            np.concatenate((distances, distances)),
        )
    else:
        raise ValueError(
            "Specify either dmax or kmax (or both)"
        )
    return pd.DataFrame(
        data={
            "Object": np.asarray(labels[indices1], dtype=io.mask_dtype),
            "Neighbor": np.asarray(labels[indices2], dtype=io.mask_dtype),
            "Distance": np.asarray(distances, dtype=np.float32),
        }
    )


def _measure_euclidean_border_distance_neighbors(
    mask: np.ndarray,
    metric: Optional[str] = None,
    dmax: Optional[float] = None,
    kmax: Optional[int] = None,
) -> pd.DataFrame:
    if metric not in (None, "euclidean"):
        raise ValueError("Metric has to be euclidean")
    labels1 = []
    labels2 = []
    distances = []
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    for label in unique_labels:
        if dmax is not None:
            ys, xs = np.nonzero(mask == label)
            dmax_int = int(np.ceil(dmax))
            ymin, ymax = np.amin(ys) - dmax_int, np.amax(ys) + dmax_int
            xmin, xmax = np.amin(xs) - dmax_int, np.amax(xs) + dmax_int
            mask_or_patch = mask[
                max(0, ymin) : min(mask.shape[0], ymax + 1),
                max(0, xmin) : min(mask.shape[1], xmax + 1),
            ]
            neighbor_labels = np.unique(mask_or_patch)
            neighbor_labels = neighbor_labels[neighbor_labels != 0]
        else:
            mask_or_patch = mask
            neighbor_labels = unique_labels
        dists = distance_transform_edt(mask_or_patch != label)
        neighbor_labels = neighbor_labels[neighbor_labels != label]
        neighbor_dists = np.array(
            [
                np.amin(dists[mask_or_patch == neighbor_label])
                for neighbor_label in neighbor_labels
            ]
        )
        if dmax is not None:
            neighbor_labels = neighbor_labels[neighbor_dists <= dmax]
            neighbor_dists = neighbor_dists[neighbor_dists <= dmax]
        if kmax is not None and len(neighbor_labels) > kmax:
            knn_indices = np.argpartition(neighbor_dists, kmax - 1)[:kmax]
            neighbor_labels = neighbor_labels[knn_indices]
            neighbor_dists = neighbor_dists[knn_indices]
        labels1 += [label] * len(neighbor_labels)
        labels2 += neighbor_labels.tolist()
        distances += neighbor_dists.tolist()
    return pd.DataFrame(
        data={
            "Object": np.asarray(labels1),
            "Neighbor": np.asarray(labels2),
            "Distance": np.asarray(distances, dtype=np.float32),
        }
    )


def _measure_euclidean_pixel_expansion_neighbors(
    mask: np.ndarray,
    metric: Optional[str] = None,
    dmax: Optional[float] = None,
    kmax: Optional[int] = None,
) -> pd.DataFrame:
    if metric not in (None, "euclidean"):
        raise ValueError(
            "Metric has to be Euclidean for pixel expansion"
        )
    if dmax is None:
        raise ValueError(
            "Maximum distance required for pixel expansion"
        )
    if kmax is not None:
        raise ValueError(
            "k-nearest neighbors is not supported by pixel expansion"
        )
    mask = _expand_mask_euclidean(mask, dmax)
    neighbors = _measure_euclidean_border_distance_neighbors(mask, dmax=1.0)
    neighbors["Distance"] = np.nan
    return neighbors


class NeighborhoodType(Enum):
    CENTROID_DISTANCE = partial(_measure_centroid_distance_neighbors)
    EUCLIDEAN_BORDER_DISTANCE = partial(_measure_euclidean_border_distance_neighbors)
    EUCLIDEAN_PIXEL_EXPANSION = partial(_measure_euclidean_pixel_expansion_neighbors)


def measure_neighbors(
    mask: np.ndarray,
    neighborhood_type: NeighborhoodType,
    metric: Optional[str] = None,
    dmax: Optional[float] = None,
    kmax: Optional[int] = None,
) -> pd.DataFrame:
    return neighborhood_type.value(mask, metric=metric, dmax=dmax, kmax=kmax)

def create_neighbors_files(
    base_dir: Union[str, Path],
    masks_folder: str = "masks_merged",
    neighbors_folder: str = "neighbors",
    neighborhood_type: Union[str, NeighborhoodType] = NeighborhoodType.EUCLIDEAN_BORDER_DISTANCE,
    metric: str = "euclidean",
    dmax: Optional[float] = 50.0,
    kmax: Optional[int] = 30,
):
    # Convert string to Enum if needed
    if isinstance(neighborhood_type, str):
        neighborhood_type = NeighborhoodType[neighborhood_type]
        
    """
    Measure neighbors from mask files and write to CSV.
    """
    base_dir = Path(base_dir)
    masks_dir = base_dir / masks_folder
    neighbors_dir = base_dir / neighbors_folder
    neighbors_dir.mkdir(exist_ok=True, parents=True)

    mask_files = sorted([f for f in masks_dir.glob("*.tiff")])
    random.shuffle(mask_files)

    for mask_path in mask_files:
        stem = mask_path.stem.replace("_mask", "")
        output_filename = f"{stem}.csv"
        output_path = neighbors_dir / output_filename

        if output_path.exists():
            print(f"Output {output_filename} exists. Skipping.")
            continue

        try:
            mask = tifffile.imread(mask_path)

            neighbors_df = measure_neighbors(
                mask,
                neighborhood_type=neighborhood_type,
                metric=metric,
                dmax=dmax,
                kmax=kmax,
            )

            neighbors_df.to_csv(output_path, index=False)
            print(f"Saved neighbors: {output_filename}")
            del mask, neighbors_df
            gc.collect()

        except Exception as e:
            print(f"Error processing {mask_path.name}: {e}")
