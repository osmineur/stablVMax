import pandas as pd
import numpy as np
import os
import random
from tifffile import imwrite, imread
from scipy.optimize import nnls
from joblib import Parallel, delayed
import warnings
from tqdm.notebook import tqdm
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm_joblib import tqdm_joblib


# Functions for pixelwise image compensation

def fast_nnls_per_pixel(spill_matrix, pixel_values):
    return nnls(spill_matrix, pixel_values)[0]


def process_row(h, tiff_stack, spill_matrix):
    H, W = tiff_stack.shape[1:]  # Get width
    return np.array([fast_nnls_per_pixel(spill_matrix, tiff_stack[:, h, w]) for w in range(W)])


def comp_cytof_tiff_parallel(tiff_stack, filename, spill_matrix, n_jobs=-1, backend="loky", position=1):
    """
    Parallel compensation for multi-image TIFFs using Joblib.
    
    Parameters:
        tiff_stack (numpy.ndarray): Multi-image TIFF (49, 1852, 1891).
        spill_matrix (pandas.DataFrame or numpy.ndarray): (49, 49) spillover matrix.
        n_jobs (int): Number of parallel jobs (-1 uses all available CPUs).
    
    Returns:
        numpy.ndarray: Compensated TIFF stack (49, 1852, 1891).
    """
    # Convert spill_matrix to NumPy array if it's a DataFrame
    if isinstance(spill_matrix, pd.DataFrame):
        spill_matrix = spill_matrix.to_numpy()
    
    # Ensure data types are compatible
    spill_matrix = spill_matrix.astype(np.float32)
    tiff_stack = tiff_stack.astype(np.float32)

    # Get dimensions
    Z, H, W = tiff_stack.shape
    with tqdm_joblib(
        tqdm(desc=f"Compensating: {filename}", total=H, unit="row", position=position, leave=False)
    ):
        compensated_rows = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(process_row)(h, tiff_stack, spill_matrix) for h in range(H)
        )
        
    # Transpose result (already allocates the big array once)
    compensated_stack = np.stack(compensated_rows, axis=1).transpose(2, 1, 0)
    del compensated_rows  # Free memory
    del tiff_stack  # Free memory

    # Convert early to float32 (halves memory traffic)
    compensated_stack = compensated_stack.astype(np.float32, copy=False)

    # Round down to 2 decimals in-place
    np.multiply(compensated_stack, 100, out=compensated_stack)
    np.floor(compensated_stack, out=compensated_stack)
    compensated_stack /= 100

    # Subtract reciprocal of signal intensity in-place
    nonzero_mask = compensated_stack != 0
    compensated_stack[nonzero_mask] -= 1.0 / compensated_stack[nonzero_mask]

    # Clip in place (avoid new array)
    np.clip(compensated_stack, a_min=0, a_max=None, out=compensated_stack)
        
    return compensated_stack


def batch_compensate_images(
    base_dir,
    spillover_filename="spillovermatrix.csv",
    overwrite=False,
    shuffle=True,
    verbose=True,
    n_jobs=-1,
    backend="loky",
    clip_max=99.9
):
    """
    Compensates TIFF images in a directory using a spillover matrix.
    Shows overall progress and per-image compensation info.
    """
    # --- Load spillover matrix ---
    spillover_path = os.path.join(base_dir, spillover_filename)
    spillovermatrix = pd.read_csv(spillover_path, index_col=0)
    panel = pd.read_csv(os.path.join(base_dir, "panel.csv"))

    # Format spillover matrix
    spillovermatrix.index = spillovermatrix.index.str.replace("Di$", "", regex=True)
    spillovermatrix.columns = spillovermatrix.columns.str.replace("Di$", "", regex=True)
    panel_channels = panel["channel"].dropna().astype(str).unique()
    spillovermatrix = spillovermatrix.loc[
        spillovermatrix.index.intersection(panel_channels),
        spillovermatrix.columns.intersection(panel_channels)
    ]
    missing_channels = set(panel_channels) - set(spillovermatrix.index)
    for ch in missing_channels:
        spillovermatrix.loc[ch] = 0
        spillovermatrix[ch] = 0
        spillovermatrix.loc[ch, ch] = 1
    spillovermatrix = spillovermatrix.loc[panel_channels, panel_channels]

    if verbose:
        print(f"Using spillover matrix with shape: {spillovermatrix.shape}")

    # --- Prepare directories ---
    img_dir = os.path.join(base_dir, "img")
    img_comp_dir = os.path.join(base_dir, "img_comp")
    os.makedirs(img_comp_dir, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.endswith(".tiff")]
    total_images = len(images)
    already_done = sum(os.path.exists(os.path.join(img_comp_dir, f)) for f in images)
    
    if verbose:
        print(f"Total images: {total_images}, already compensated: {already_done}")
    
    if shuffle:
        random.shuffle(images)

    with tqdm(total=total_images, desc="Compensating images", position=0) as global_bar:
        if not overwrite:
            global_bar.update(already_done)

        for filename in images:
            in_path = os.path.join(img_dir, filename)
            out_path = os.path.join(img_comp_dir, filename)

            if not overwrite and os.path.exists(out_path):
                img=None
                img_comp=None
                continue

            img = imread(in_path)
            img_comp = comp_cytof_tiff_parallel(
                img, filename, spillovermatrix, n_jobs=n_jobs, backend=backend, position=1
            )

            # --- Clip per channel ---
            for c in range(img_comp.shape[0]):
                nonzero_pixels = img_comp[c][img_comp[c] > 0]
                if len(nonzero_pixels) > 0:
                    clip_val = np.percentile(nonzero_pixels, clip_max)
                    img_comp[c] = np.clip(img_comp[c], 0, clip_val)

            # --- Global clipping across channels ---
            summed = img_comp.sum(axis=0)
            nonzero_summed = summed[summed > 0]
            if len(nonzero_summed) > 0:
                sum_clip_val = np.percentile(nonzero_summed, clip_max)
                mask = summed > sum_clip_val
                if np.any(mask):
                    scale = sum_clip_val / summed[mask]
                    img_comp[:, mask] = img_comp[:, mask] * scale

            imwrite(out_path, img_comp, compression="zlib", photometric="minisblack")
        
        del img, img_comp
