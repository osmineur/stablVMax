import warnings
import os
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from typing import Optional, Union, List
from pathlib import Path
from readimc import MCDFile  # adjust if your import path is different
import shutil


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


### Create image list from MCD files ###

def create_image_list_csv(base_dir):
    raw_dir = Path(base_dir) / "raw"
    mcd_files = list(raw_dir.glob("*.mcd"))
    image_info_list = []

    for mcd_file in mcd_files:
        with MCDFile(mcd_file) as mcd:
            for acquisition in mcd.acquisitions:
                sample = mcd_file.stem
                roi = acquisition.description.split('_')[-1] if acquisition.description else f"ROI{acquisition.id}"
                img_filename = f"{sample}_{roi}.tiff"

                image_info = {
                    "sample": sample,
                    "roi": roi,
                    "mcd_file": str(mcd_file),
                    "acquisition_id": acquisition.id,
                    "description": acquisition.description,
                    "width": acquisition.width,
                    "height": acquisition.height,
                    "img_filename": img_filename,
                }
                image_info_list.append(image_info)

    image_info_df = pd.DataFrame(image_info_list)
    image_info_df.to_csv(os.path.join(base_dir, "images.csv"), index=False)
    return image_info_df



### Panel creation from MCD files ###

def create_panels_from_mcd_file(mcd_file: Union[str, Path]) -> List[pd.DataFrame]:
    """Extract channel panel(s) from a single MCD file."""
    panels = []
    with MCDFile(mcd_file) as f:
        for slide in f.slides:
            for acquisition in slide.acquisitions:
                panel = pd.DataFrame({
                    "channel": acquisition.channel_names,
                    "marker": [label.split("_")[1] if len(label.split("_")) > 1 else None for label in acquisition.channel_labels],
                    "name": acquisition.channel_labels #[f"{n}_{l}" for n, l in zip(acquisition.channel_names, acquisition.channel_labels)]
                })
                panels.append(panel)
    return panels


def create_panel_from_mcd_files(mcd_files: List[Union[str, Path]]) -> pd.DataFrame:
    """Combine panels from multiple MCD files into one unique panel."""
    panels = []
    for mcd_file in mcd_files:
        panels += create_panels_from_mcd_file(mcd_file)

    # Combine and drop duplicates
    panel = pd.concat(panels, ignore_index=True)
    panel.drop_duplicates(inplace=True, ignore_index=True)

    # Sort numerically by channel (ignores non-numeric parts)
    panel = panel.sort_values(
        "channel",
        key=lambda s: pd.to_numeric(s.str.replace("[^0-9]", "", regex=True)),
    ).reset_index(drop=True)

    # Add index column
    panel.insert(0, "index", range(len(panel)))

    return panel


def create_panel_csv(base_dir: str) -> pd.DataFrame:
    """Create a panel.csv from MCD files in base_dir/raw."""
    raw_dir = Path(base_dir) / "raw"
    mcd_files = list(raw_dir.glob("*.mcd"))

    print(f"Found {len(mcd_files)} MCD files.")

    panel = create_panel_from_mcd_files(mcd_files)

    out_path = Path(base_dir) / "panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"Saved panel to: {out_path}")

    return panel


### Conversion to TIFFs ###

def filter_hot_pixels(img: np.ndarray, thres: float) -> np.ndarray:
    """Replace hot pixels that exceed their neighbors by threshold."""
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)

def preprocess_image(img: np.ndarray, hpf: Optional[float] = None, dtype=np.float32) -> np.ndarray:
    """
    Preprocess image:
    - convert to float32 internally
    - optionally filter hot pixels
    - return as target dtype (default: float32)
    """
    img = img.astype(np.float32, copy=False)
    if hpf is not None:
        img = filter_hot_pixels(img, hpf)
    return img.astype(dtype)


def preprocess_mcd_images_from_disk(
    mcd_files,
    hpf=None,
    strict=False,
):
    """
    Generator: yield (Path, acquisition, image) for each acquisition in MCD files.
    No txt/zipped support, only raw MCD. Always loads all channels.
    """
    for mcd_file in sorted(mcd_files, key=lambda f: Path(f).stem, reverse=True):
        with MCDFile(mcd_file) as f_mcd:
            for slide in f_mcd.slides:
                for acquisition in slide.acquisitions:
                    img = f_mcd.read_acquisition(acquisition, strict=strict)
                    img = preprocess_image(img, hpf=hpf)
                    yield Path(mcd_file), acquisition, img
                    del img

def trim_zeros_from_image(img):
    """
    Trim away rows/columns where all channels are zero.
    img shape: (C, H, W)
    """
    # Collapse channels to detect "any signal"
    nonzero_mask = np.any(img != 0, axis=0)  # shape (H, W)

    # Keep rows and cols that contain any nonzero pixel
    row_mask = np.any(nonzero_mask, axis=1)
    col_mask = np.any(nonzero_mask, axis=0)

    trimmed = img[:, row_mask][:, :, col_mask]
    return trimmed

def convert_mcd_to_tiff(
    base_dir,
    hpf=50,
    mcd_filename: Optional[Union[str, Path]] = None,
    roi_match: Optional[str] = None,
    strict=False,
    image_names="roi" # alternative "long"
):
    """
    Convert MCD file(s) to TIFFs with optional filtering.
    
    Args:
        base_dir: project base directory (expects raw/ inside)
        hpf: hot pixel filter threshold
        mcd_filename: restrict to a specific MCD file (str/Path)
        roi_match: restrict to acquisitions whose description contains this string
        strict: pass-through to readimc (default False)
    """
    raw_dir = Path(base_dir) / "raw"
    img_dir = Path(base_dir) / "img"
    img_dir.mkdir(exist_ok=True)

    # Pick files
    if mcd_filename:
        mcd_files = [raw_dir / Path(mcd_filename)]
    else:
        mcd_files = list(raw_dir.glob("*.mcd"))

    print(f"Found {len(mcd_files)} MCD file(s).")

    for mcd_file, acquisition, img in preprocess_mcd_images_from_disk(mcd_files, hpf=hpf, strict=strict):
        # --- filter ROI by description if requested ---
        if roi_match and (not acquisition.description or roi_match not in acquisition.description):
            continue

        sample = Path(mcd_file).stem
        roi = acquisition.description.split("_")[-1] if acquisition.description else f"ROI{acquisition.id}"
        if image_names == "roi":
            img_filename = f"{roi}.tiff"
        else:
            img_filename = f"{sample}_{roi}.tiff"
        img_path = img_dir / img_filename

        img = trim_zeros_from_image(img)  # optional trimming step

        tifffile.imwrite(img_path, img, compression="zlib", photometric="minisblack")
        print(f"Saved {img_path}")



### Stitching images ###

def stitch_images(
    base_dir,
    img_top_name,
    img_bottom_name,
    new_img_name,
    img_folder="img",
    visualize_channel_index=None,
):
    img_dir = os.path.join(base_dir, img_folder)
    img_archive_dir = os.path.join(img_dir, "img_archive")
    os.makedirs(img_archive_dir, exist_ok=True)  # create subfolder

    top_path = os.path.join(img_dir, img_top_name)
    bottom_path = os.path.join(img_dir, img_bottom_name)
    top_prestitch_path = os.path.join(img_archive_dir, img_top_name.replace(".tiff", "_prestitch.tiff"))
    bottom_prestitch_path = os.path.join(img_archive_dir, img_bottom_name.replace(".tiff", "_prestitch.tiff"))

    if os.path.exists(top_prestitch_path):
        print(f"Top image {img_top_name} (prestitch) already in archive folder")
        return
    if os.path.exists(bottom_prestitch_path):
        print(f"Bottom image {img_bottom_name} (prestitch) already in archive folder")
        return

    # --- Check top image ---
    if os.path.exists(top_path):
        pass
    else:
        print(f"Top image {img_top_name} not found in {img_dir}")
        return

    # --- Check bottom image ---
    if os.path.exists(bottom_path):
        pass
    else:
        print(f"Bottom image {img_bottom_name} not found in {img_dir}")
        return

    top = tifffile.imread(top_path)
    bottom = tifffile.imread(bottom_path)

    # --- Trim zeros from bottom of top ---
    row_sums_top = np.sum(top, axis=(0, 2))
    nonzero_rows_top = np.where(row_sums_top != 0)[0]
    last_nonzero_top = nonzero_rows_top.max() if len(nonzero_rows_top) > 0 else top.shape[1] - 1
    top_trimmed = top[:, :last_nonzero_top+1, :]

    # --- Trim zeros from top of bottom ---
    row_sums_bottom = np.sum(bottom, axis=(0, 2))
    nonzero_rows_bottom = np.where(row_sums_bottom != 0)[0]
    first_nonzero_bottom = nonzero_rows_bottom.min() if len(nonzero_rows_bottom) > 0 else 0
    bottom_trimmed = bottom[:, first_nonzero_bottom:, :]

    # --- Stitch ---
    stitched = np.concatenate([top_trimmed, bottom_trimmed], axis=1)

    # --- Optional visualization ---
    if visualize_channel_index is not None:
        img = np.arcsinh(stitched[visualize_channel_index])
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")
        plt.title(f"Stitched image - Channel index {visualize_channel_index}")
        plt.axis("off")
        plt.show()

    # --- Save stitched image ---
    new_img_path = os.path.join(img_dir, new_img_name)
    tifffile.imwrite(new_img_path, stitched, compression="zlib", photometric="minisblack")

    # --- Move original top/bottom images into img_archive with _prestitch suffix ---
    top_prestitch = os.path.join(img_archive_dir, f"{Path(img_top_name).stem}_prestitch.tiff")
    bottom_prestitch = os.path.join(img_archive_dir, f"{Path(img_bottom_name).stem}_prestitch.tiff")
    shutil.move(top_path, top_prestitch)
    shutil.move(bottom_path, bottom_prestitch)

    print(f"Stitched image saved to {new_img_path}")
    print(f"Original images moved to {img_archive_dir} with '_prestitch' suffix")

    return stitched


def archive_images(base_dir, img_folder, filenames, archive_folder="img_archive"):
    img_dir = os.path.join(base_dir, img_folder)
    archive_path = os.path.join(img_dir, archive_folder)
    os.makedirs(archive_path, exist_ok=True)

    for f in filenames:
        f_path = os.path.join(img_dir, f)
        dest_path = os.path.join(archive_path, f)

        if os.path.exists(f_path):
            shutil.move(f_path, dest_path)
            print(f"Moved {f} to {archive_folder}")
        elif os.path.exists(dest_path):
            print(f"{f} is already archived in archive folder")
        else:
            print(f"{f} not found in {img_dir}")