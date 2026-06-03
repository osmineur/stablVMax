import os
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from skimage import io, img_as_ubyte
from skimage.segmentation import find_boundaries
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colors as mcolors
from matplotlib import cm


def adjust_contrast(image, upper_bound=98, lower_bound=2):
    image = image.astype(np.float32)
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(nonzero, [lower_bound, upper_bound])
    if high <= low:
        return np.zeros_like(image)
    norm = (image - low) / (high - low)
    return np.clip(norm, 0, 1)


def process_image_crop_and_scale(
    image: np.ndarray,
    roi: tuple = None,        # (x_min, y_min, width, height)
    crop: bool = False,
    scale_bar=None,           # int or tuple(length_px, thickness_px, color)
    scale_text: bool = False,
    text_offset: int = 10
) -> np.ndarray:
    """
    Apply ROI cropping/boxing and optional scale bar drawing.

    Parameters
    ----------
    image : np.ndarray
        RGB image (float [0,1] or uint8 [0,255]).
    roi : tuple, optional
        (x_min, y_min, width, height) of ROI in pixels.
    crop : bool, default False
        If True, return only the cropped ROI.
        If False, draw a white box around the ROI but keep the full image.
    scale_bar : int or tuple, optional
        Either an int (length in pixels, default thickness=5, white),
        or (length_px, thickness_px, color) where color is RGB tuple or hex string.
    scale_text : bool, default False
        If True, draw scale bar length as text above the bar.
    text_offset : int, default 10
        Pixel offset of text from scale bar.

    Returns
    -------
    img : np.ndarray
        Processed image with ROI box/crop and scale bar.
    """
    img = image.copy()

    # Determine dtype & scale factor
    if np.issubdtype(img.dtype, np.floating):
        scale_factor = 1.0
    elif np.issubdtype(img.dtype, np.integer):
        scale_factor = 255.0
    else:
        raise ValueError(f"Unsupported image dtype {img.dtype}")

    # --- Handle ROI ---
    if roi is not None:
        x_min, y_min, w, h_roi = roi
        x_max, y_max = x_min + w, y_min + h_roi

        if crop:
            img = img[y_min:y_max, x_min:x_max, :]
        else:
            # Draw 3-pixel-wide white rectangle
            thickness = 5
            color_box = 1.0 if scale_factor==1.0 else 255
            img[y_min:y_max, x_min:x_min+thickness, :] = color_box
            img[y_min:y_max, x_max-thickness:x_max, :] = color_box
            img[y_min:y_min+thickness, x_min:x_max, :] = color_box
            img[y_max-thickness:y_max, x_min:x_max, :] = color_box

    # --- Handle Scale Bar ---
    if scale_bar is not None:
        # Default scale bar
        if isinstance(scale_bar, int):
            length = scale_bar
            thickness = 10
            color = (1.0,1.0,1.0) if scale_factor==1.0 else (255,255,255)
        elif isinstance(scale_bar, (tuple, list)) and len(scale_bar) == 3:
            length, thickness, color = scale_bar
            # Convert hex string if needed
            if isinstance(color, str) and color.startswith("#") and len(color)==7:
                color = tuple(int(color[i:i+2],16) for i in (1,3,5))
                if scale_factor==1.0:
                    color = tuple(c/255 for c in color)
            elif isinstance(color, (tuple,list)):
                color = tuple(c/255 if scale_factor==1.0 else c for c in color)
            else:
                raise ValueError(f"Unsupported color format: {color}")
        else:
            raise ValueError("scale_bar must be int or tuple/list (length, thickness, color)")

        # Image size
        h_img, w_img = img.shape[:2]

        # Bottom-left corner of bar
        x_start = 50
        y_start = h_img - 50
        x_end = x_start + length
        y_end = y_start - thickness

        # Draw rectangle
        img[y_end:y_start, x_start:x_end, :] = color

# Draw scale text
        if scale_text:
            
            font_scale = 1.0
            thickness_text = 2
            text = str(length)
            
            # Convert img to uint8 PIL Image
            if scale_factor == 1.0:
                tmp = (img * 255).astype(np.uint8)
            else:
                tmp = img.copy().astype(np.uint8)
            
            pil_img = Image.fromarray(tmp)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to load a default font, fall back to PIL default
            try:
                pil_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=int(20 * font_scale))
            except:
                pil_font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), text, font=pil_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            text_x = x_start + (length - tw) // 2
            text_y = y_end - text_offset
            if text_y - th < 0:
                text_y = y_end + text_offset  # fallback below bar
            
            # Build PIL color tuple
            if scale_factor == 1.0:
                pil_color = tuple(int(c * 255) for c in color)
            else:
                pil_color = tuple(int(c) for c in color)
            
            draw.text((text_x, text_y), text, font=pil_font, fill=pil_color)
            
            # Convert back to original format
            tmp = np.array(pil_img)
            if scale_factor == 1.0:
                img = tmp.astype(np.float32) / 255
            else:
                img = tmp

    return img



def create_multicolor_image(
    adata: AnnData,
    channels: list,
    image_name: str,
    base_dir: str,
    output_dir: str = None,
    img_dir: str = None,
    title: str = "Multi-color Image",
    channel_colors: dict = None,
    contrast_bounds=(10, 90),
    roi: tuple = None,
    crop: bool = False,
    scale_bar: tuple = None
):
    if len(channels) > 4:
        raise ValueError("A maximum of 4 channels can be visualized.")

    if img_dir:
        img_dir = os.path.join(base_dir, img_dir)
    else:
        img_dir = os.path.join(base_dir, "img_comp")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(base_dir, "img")
    default_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]

    if channel_colors is None:
        channel_colors = {ch: default_colors[i % 4] for i, ch in enumerate(channels)}
    else:
        for i, ch in enumerate(channels):
            if ch not in channel_colors:
                channel_colors[ch] = default_colors[i % 4]

    channel_colors = {ch: mcolors.to_rgb(c) for ch, c in channel_colors.items()}

    img_file = next((f for f in os.listdir(img_dir) if image_name in f), None)
    if not img_file:
        print(f"[!] No TIFF file found for image '{image_name}' in {img_dir}")
        return
    img_path = os.path.join(img_dir, img_file)
    img = io.imread(img_path)
    if len(img.shape) != 3:
        print(f"[!] Unexpected TIFF shape: {img.shape} (expected 3D stack).")
        return

    rgb_image = np.zeros((*img.shape[1:], 3), dtype=np.float32)
    for ch in channels:
        if ch not in adata.var.index:
            print(f"[!] Channel '{ch}' not found in adata.var.index.")
            continue
        ch_idx = adata.var.index.get_loc(ch)
        if ch_idx >= img.shape[0]:
            print(f"[!] Channel index {ch_idx} for '{ch}' exceeds TIFF shape.")
            continue
        ch_data = img[ch_idx]
        lower_bound, upper_bound = contrast_bounds
        ch_norm = adjust_contrast(ch_data, upper_bound=upper_bound, lower_bound=lower_bound)
        color = np.array(channel_colors[ch])
        for i in range(3):
            rgb_image[..., i] += ch_norm * color[i]

    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image = process_image_crop_and_scale(rgb_image, roi=roi, crop=crop, scale_bar=scale_bar)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.gca().set_aspect("equal") 
    plt.show()

    if output_dir:
        os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)
        out_path = os.path.join(base_dir, output_dir, f"{image_name}_multicolor.tiff")
        Image.fromarray((rgb_image*255).astype(np.uint8)).save(out_path)
        print(f"[✓] Saved TIFF to {out_path}")


def mask_colored_by_category(
    adata: AnnData,
    image_name: str,
    base_dir: str,
    category_column: str = "scyan_pop",
    mask_subdir: str = "masks_merged",
    output_dir: str = None,
    custom_palette: dict = None,
    roi: tuple = None,
    crop: bool = False,
    scale_bar: tuple = None,
    background_color: str = "black"
):
    image_adata = adata[adata.obs['Image'] == image_name]
    if image_adata.obs.empty:
        print(f"[!] No data found for image '{image_name}'.")
        return

    mask_dir = os.path.join(base_dir, mask_subdir)
    mask_file = next((f for f in os.listdir(mask_dir) if image_name in f), None)
    if not mask_file:
        print(f"[!] No mask file found for image '{image_name}' in {mask_dir}.")
        return
    mask = io.imread(os.path.join(mask_dir, mask_file))

    if category_column not in image_adata.obs.columns:
        print(f"[!] Column '{category_column}' not found in adata.obs.")
        return
    unique_categories = sorted(image_adata.obs[category_column].dropna().unique())

    color_map = {}
    n_cats = len(unique_categories)
    
    if isinstance(custom_palette, dict):
        # Use dictionary mapping
        cmap = plt.get_cmap("tab20")  # fallback for missing categories
        for i, cat in enumerate(unique_categories):
            if cat in custom_palette:
                color_map[cat] = mcolors.to_rgb(custom_palette[cat])
            else:
                print(f"{cat} not in custom_palette dict → using default tab20")
                color_map[cat] = cmap(i % cmap.N)[:3]

    elif isinstance(custom_palette, mcolors.Colormap):
        # Sample colors evenly across the colormap
        for i, cat in enumerate(unique_categories):
            color_map[cat] = custom_palette(i / max(1, n_cats - 1))[:3]

    else:
        # Default tab20
        cmap = plt.get_cmap("tab20")
        for i, cat in enumerate(unique_categories):
            color_map[cat] = cmap(i % cmap.N)[:3]

    # colored_mask = np.zeros((*mask.shape, 3), dtype=np.float32)
    colored_mask = np.ones((*mask.shape, 3), dtype=np.float32) * np.array(mcolors.to_rgb(background_color), dtype=np.float32)
    for cat in unique_categories:
        labels = (
            image_adata.obs[image_adata.obs[category_column] == cat]
            .index.str.split('_')
            .str[-1]
            .astype(int)
        )
        binary_mask = np.isin(mask, labels)
        colored_mask[binary_mask] = color_map[cat]  # set directly instead of +=

    borders = find_boundaries(mask, connectivity=1, mode='thick')
    colored_mask[borders] = [0, 0, 0]
    colored_mask = np.clip(colored_mask, 0, 1)
    colored_mask = process_image_crop_and_scale(colored_mask, roi=roi, crop=crop, scale_bar=scale_bar)
    colored_mask = np.clip(colored_mask, 0, 1)

    if output_dir:
        os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)
        tiff_path = os.path.join(base_dir, output_dir, f"{image_name}_colored_by_{category_column}.tiff")
        io.imsave(tiff_path, img_as_ubyte(colored_mask))
        print(f"[✓] Saved TIFF to {tiff_path}")

    plt.figure(figsize=(12, 10))
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.title(f"{image_name} colored by '{category_column}'")
   
    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color=color_map[cat], markersize=10, linestyle='', label=cat)
        for cat in unique_categories
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=category_column)
    plt.tight_layout()
    plt.show()


def mask_colored_by_continuous(
    adata: AnnData,
    image_name: str,
    base_dir: str,
    value_column: str,
    mask_subdir: str = "masks_merged",
    output_dir: str = None,
    cmap_name: str = "viridis",
    clip_percentiles: tuple = (2, 98),
    roi: tuple = None,
    crop: bool = False,
    scale_bar: tuple = None
):
    image_adata = adata[adata.obs["Image"] == image_name]
    if image_adata.shape[0] == 0:
        print(f"[!] No data found for image '{image_name}'.")
        return

    if value_column in image_adata.obs.columns:
        values = image_adata.obs[value_column].values
    elif value_column in adata.var_names:
        var_idx = adata.var_names.get_loc(value_column)
        values = image_adata.layers['counts'][:, var_idx].toarray().flatten() if hasattr(image_adata.layers['counts'], "toarray") else image_adata.layers['counts'][:, var_idx]
        values = np.arcsinh(values) 
    else:
        print(f"[!] Column '{value_column}' not found.")
        return

    vmin, vmax = np.nanpercentile(values, clip_percentiles)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap_name)
    color_values = colormap(norm(values))[:, :3]

    mask_dir = os.path.join(base_dir, mask_subdir)
    mask_file = next((f for f in os.listdir(mask_dir) if image_name in f), None)
    if not mask_file:
        print(f"[!] No mask file found for image '{image_name}' in {mask_dir}.")
        return
    mask = io.imread(os.path.join(mask_dir, mask_file))

    try:
        cell_ids = image_adata.obs.index.str.split("_").str[-1].astype(int).values
    except Exception as e:
        print("[!] Could not extract numeric cell IDs.")
        print(e)
        return

    max_label = mask.max()
    color_lookup = np.zeros((max_label + 1, 3), dtype=np.float32)
    for cell_id, color in zip(cell_ids, color_values):
        if 0 < cell_id <= max_label:
            color_lookup[cell_id] = color

    colored_mask = color_lookup[mask]
    borders = find_boundaries(mask, connectivity=1, mode='inner')
    colored_mask[borders] = [0, 0, 0]
    colored_mask = np.clip(colored_mask, 0, 1)
    colored_mask = process_image_crop_and_scale(colored_mask, roi=roi, crop=crop, scale_bar=scale_bar)

    if output_dir:
        os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)
        out_path = os.path.join(base_dir, output_dir, f"{image_name}_colored_by_{value_column}.tiff")
        io.imsave(out_path, img_as_ubyte(colored_mask))
        print(f"[✓] Saved TIFF to {out_path}")

    plt.figure(figsize=(10, 10))
    im = plt.imshow(colored_mask, interpolation="none")
    plt.axis("off")
    plt.title(f"{image_name} colored by '{value_column}'")
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(im, shrink=0.8, pad=0.02, label=value_column)
    plt.gca().set_aspect("equal") 
    plt.tight_layout()
    plt.show()