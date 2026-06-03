import os
import time
import random
import numpy as np
import tifffile
from scipy.ndimage import label


def _load_masks(mask_file, masks_vessel_dir, masks_muscle_dir, masks_cells_dir):
    vessel_mask = tifffile.imread(os.path.join(masks_vessel_dir, mask_file))
    muscle_mask = tifffile.imread(os.path.join(masks_muscle_dir, mask_file))
    cells_mask = tifffile.imread(os.path.join(masks_cells_dir, mask_file))
    return vessel_mask, muscle_mask, cells_mask

def simple_merge(primary_mask, secondary_mask, overlap_threshold=0.75):
    """
    Merge secondary_mask into primary_mask based on bidirectional overlap.
    Preserves primary continuity. Secondary objects are always fully added, never broken.
    """
    merged_mask = primary_mask.copy()
    current_max_id = merged_mask.max()

    # Precompute areas
    primary_dict = get_object_areas(primary_mask)
    secondary_dict = get_object_areas(secondary_mask)

    # Compute overlap matrix
    overlap_matrix = _compute_overlap_matrix(primary_mask, secondary_mask)

    for secondary_id in range(1, overlap_matrix.shape[0]):
        overlaps = overlap_matrix[secondary_id]
        obj_area = secondary_dict.get(secondary_id, 0)
        if obj_area == 0:
            continue

        obj_mask = (secondary_mask == secondary_id)
        strong_ids = get_strong_overlap_ids(overlaps, obj_area, primary_dict, overlap_threshold)

        if not strong_ids:
            # No strong overlap: Add full object safely (only to background), ensuring continuity
            merged_mask, current_max_id = assign_secondary_object_safely(
                merged_mask, secondary_mask, secondary_id, current_max_id
            )
        elif len(strong_ids) == 1:
            # One strong overlap: merge into that primary
            merged_mask, current_max_id = safe_merge_single(
                primary_mask, merged_mask, strong_ids[0], obj_mask, current_max_id
            )
        else:
            # Multiple strong overlaps: union all involved regions and assign new ID(s)
            merged_mask, current_max_id = handle_multiple_strong_overlaps(
                secondary_mask, primary_mask, merged_mask, obj_mask, strong_ids, current_max_id
            )

    return merged_mask

def get_object_areas(mask):
    ids, areas = np.unique(mask, return_counts=True)
    return {i: a for i, a in zip(ids, areas) if i != 0}


def get_strong_overlap_ids(overlaps, secondary_area, primary_dict, threshold):
    strong_ids = []
    for primary_id in range(1, len(overlaps)):
        overlap = overlaps[primary_id]
        primary_area = primary_dict.get(primary_id, 0)
        if primary_area == 0 or overlap == 0:
            continue

        secondary_ratio = overlap / secondary_area
        primary_ratio = overlap / primary_area

        if secondary_ratio > threshold or primary_ratio > threshold:
            strong_ids.append(primary_id)

    return strong_ids


def assign_secondary_object_safely(merged_mask, secondary_mask, secondary_id, current_max_id):
    obj_mask = (secondary_mask == secondary_id)
    target_pixels = obj_mask & (merged_mask == 0)

    if not np.any(target_pixels):
        return merged_mask, current_max_id

    labeled, num = label(target_pixels, )
    for i in range(1, num + 1):
        comp = (labeled == i)
        if np.count_nonzero(comp) > 0:
            current_max_id += 1
            merged_mask[comp] = current_max_id

    return merged_mask, current_max_id


def safe_merge_single(primary_mask, merged_mask, primary_id, obj_mask, current_max_id):
    union_mask = ((merged_mask == 0) | (merged_mask == primary_id)) & obj_mask
    full_union_mask = (primary_mask == primary_id) | union_mask

    # Remove original from merged_mask
    original_primary = (merged_mask == primary_id)
    merged_mask[original_primary] = 0

    # Check for continuity
    labeled, num = label(full_union_mask)
    if num == 1:
        merged_mask[full_union_mask] = primary_id
    else:
        for i in range(1, num + 1):
            comp = (labeled == i)
            if np.count_nonzero(comp) > 0:
                current_max_id += 1
                merged_mask[comp] = current_max_id

    return merged_mask, current_max_id


def handle_multiple_strong_overlaps(secondary_mask, primary_mask, merged_mask, obj_mask, strong_ids, current_max_id):
    union_mask = obj_mask.copy()

    for pid in strong_ids:
        union_mask |= (primary_mask == pid)
        merged_mask[merged_mask == pid] = 0  # Clear all involved primary IDs

    target_pixels = union_mask & (merged_mask == 0)
    if not np.any(target_pixels):
        return merged_mask, current_max_id

    # Handle disconnected union
    labeled_union, num = label(target_pixels)
    for i in range(1, num + 1):
        comp = (labeled_union == i)
        if np.count_nonzero(comp) > 0:
            current_max_id += 1
            merged_mask[comp] = current_max_id

    return merged_mask, current_max_id


def _compute_overlap_matrix(primary_mask, secondary_mask):
    max_primary = primary_mask.max()
    max_secondary = secondary_mask.max()
    overlap_matrix = np.zeros((max_secondary + 1, max_primary + 1), dtype=np.int32)

    combined = secondary_mask.astype(np.int64) * (max_primary + 1) + primary_mask
    vals, counts = np.unique(combined, return_counts=True)

    for val, count in zip(vals, counts):
        sec_id = val // (max_primary + 1)
        prim_id = val % (max_primary + 1)
        overlap_matrix[sec_id, prim_id] = count

    return overlap_matrix


def _reindex_labels_fast(mask, start_from=1):
    """
    Reindex mask labels to be continuous starting from `start_from` using vectorized array mapping.
    Assumes labels are positive integers and not too large.
    """
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # skip background

    max_id = unique_ids.max()
    mapping_array = np.zeros(max_id + 1, dtype=mask.dtype)
    mapping_array[unique_ids] = np.arange(start_from, start_from + len(unique_ids), dtype=mask.dtype)

    # Use mapping array to transform mask
    reindexed_mask = mapping_array[mask]
    return reindexed_mask


def remove_small_objects(mask, size_threshold):
    labels, counts = np.unique(mask, return_counts=True)
    remove_ids = labels[(labels != 0) & (counts < size_threshold)]
    mask_filtered = mask.copy()
    for rid in remove_ids:
        mask_filtered[mask_filtered == rid] = 0
    return mask_filtered



def combine_masks(base_dir, masks_cells_folder, masks_vessel_folder, masks_muscle_folder, masks_merged_folder, size_threshold=12, overwrite=False):
    
    """
    Combines vessel, muscle, and cells masks into a single mask and saves the result.

    Parameters:
    - masks_cells_folder: str, folder of cells mask directory
    - masks_vessel_folder: str, folder of vessel mask directory
    - masks_muscle_folder: str, folder of muscle mask directory
    - masks_merged_folder: str, folder to save the merged mask
    - size_threshold: int, minimum size for cells
    - overwrite: bool, whether to overwrite existing output
    """

    masks_cells_dir = os.path.join(base_dir, masks_cells_folder)
    masks_vessel_dir = os.path.join(base_dir, masks_vessel_folder)
    masks_muscle_dir = os.path.join(base_dir, masks_muscle_folder)
    masks_merged_dir = os.path.join(base_dir, masks_merged_folder)
    os.makedirs(masks_merged_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(masks_cells_dir) ]
    random.shuffle(mask_files)

    for mask_file in mask_files:
        
        output_path = os.path.join(masks_merged_dir, mask_file)
        if os.path.exists(output_path) and not overwrite:
            print(f"Combined mask for {mask_file} already exists. Skipping.")
            continue

        print(f"First merge {mask_file}...")
        starttime = time.time()

        vessel_mask, muscle_mask, cell_mask = _load_masks(mask_file, masks_vessel_dir, masks_muscle_dir, masks_cells_dir)

        m1 = simple_merge(primary_mask = vessel_mask, secondary_mask = muscle_mask)
        m1 = _reindex_labels_fast(m1)

        print(f'Second merge. {time.time()-starttime}')
        m2 = simple_merge(primary_mask = cell_mask, secondary_mask = m1)
        print(f'Removing small objects. {time.time()-starttime}')
        m2 = remove_small_objects(m2, size_threshold=size_threshold)
        print(f'Reindexing. {time.time()-starttime}')
        m2 = _reindex_labels_fast(m2)

        tifffile.imwrite(output_path, m2)

        endtime = time.time()
        print(f"Finished combining {mask_file} in {endtime - starttime:.2f} seconds")
