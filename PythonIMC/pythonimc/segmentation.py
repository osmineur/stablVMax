import os
import pandas as pd
import numpy as np
import tifffile
import gc
import random
import steinbock.io as io
import imageio.v3 as iio

from pathlib import Path
from scipy.ndimage import convolve


def adjust_contrast(image, upper_bound=98, lower_bound=2):
    image = image.astype(np.float32)

    nonzero_pixels = image[image > 0]
    if nonzero_pixels.size == 0:
        return np.zeros_like(image)

    lower, upper = np.percentile(nonzero_pixels, [lower_bound, upper_bound])
    threshold = upper / (lower * 10)

    image[image < threshold] = 0
    image = np.clip(image, 0, upper)

    return image


def get_channel_indices(base_dir, markers):
    panel = pd.read_csv(os.path.join(base_dir, "panel.csv"))
    available_cols = [c for c in ["name", "marker", "channel"] if c in panel.columns]

    if isinstance(markers, str):
        markers = [markers]

    indices = []
    not_found = []

    for marker in markers:
        found_idx = None
        for col in available_cols:
            match = panel.index[panel[col] == marker]
            if not match.empty:
                found_idx = match[0]
                break
        if found_idx is not None:
            indices.append(found_idx)
        else:
            not_found.append(marker)

    if not_found:
        raise ValueError(f"Markers not found in panel: {not_found}")

    return indices


def subset_img_channels(base_dir, img_path, markers):
    channels = get_channel_indices(base_dir, markers)
    img = tifffile.imread(img_path)
    if img.ndim == 3 and img.shape[0] >= 2:
        print(f"Processing: {img_path}")
        img = img[channels, :, :]
        filtered_img = np.stack(img, axis=0).astype(np.float32)
        del img
        return filtered_img
    else:
        print(f"Image {img_path} does not have the expected shape with 2 or more channels.")
        return None


def preprocess_img(img, upper_bound=98, lower_bound=2):
    if img is None:
        return
    img = [adjust_contrast(img[i, ...], upper_bound, lower_bound) for i in range(img.shape[0])]
    img = np.stack(img, axis=0).astype(np.float32)
    img = np.max(img, axis=0)
    return img



def segment_imgs_mesmer(base_dir,
                        input_folder='img_comp',
                        output_folder='masks',
                        markers=['DNA1'],
                        membrane_markers=None,
                        postprocess_kwargs=None,
                        upper_bound=98,
                        lower_bound=2,
                        image_mpp=0.5,
                        overwrite=False):
    """
    Segment images using Mesmer (DeepCell / TensorFlow).

    Requires Python < 3.11 and the deepcell package.
    Use the 'pythonimc-mesmer' conda environment.

    Parameters
    ----------
    postprocess_kwargs : dict or None
        Passed to Mesmer.predict() as postprocess_kwargs_whole_cell.
        Example: {'maxima_threshold': 0.075, 'interior_threshold': 0.2,
                  'interior_smooth': 2, 'small_objects_threshold': 60}
    """
    from deepcell.applications import Mesmer
    mesmer_model = Mesmer()

    img_dir   = os.path.join(base_dir, input_folder)
    masks_dir = os.path.join(base_dir, output_folder)
    os.makedirs(masks_dir, exist_ok=True)

    images = [x for x in io.list_image_files(img_dir)]
    if not images:
        print(f"No images found in {img_dir}.")
        return
    print(f"Found {len(images)} image(s) in {img_dir}")
    random.shuffle(images)

    for img_path in images:
        img_name  = os.path.basename(img_path)
        mask_path = os.path.join(masks_dir, img_name)

        if os.path.exists(mask_path) and not overwrite:
            print(f"Mask for {img_name} already exists. Skipping.")
            continue

        raw_nuclear = subset_img_channels(base_dir, img_path, markers=markers)
        if raw_nuclear is None:
            print(f"Skipping {img_name}: invalid image.")
            continue
        nuclear_img = np.max(raw_nuclear, axis=0).astype(np.float32)

        if membrane_markers is not None:
            raw_membrane = subset_img_channels(base_dir, img_path, markers=membrane_markers)
            membrane_img = np.max(raw_membrane, axis=0).astype(np.float32) if raw_membrane is not None else np.zeros_like(nuclear_img)
        else:
            membrane_img = np.zeros_like(nuclear_img)

        # Mesmer expects (batch, H, W, 2): channel 0 = nuclear, channel 1 = membrane
        # DeepCell handles normalisation internally — do not preprocess here.
        seg_input = np.stack([nuclear_img, membrane_img], axis=-1)[np.newaxis]

        labels = mesmer_model.predict(
            seg_input,
            image_mpp=image_mpp,
            batch_size=1,
            compartment='whole-cell',
            postprocess_kwargs_whole_cell=postprocess_kwargs or {},
        )
        labels = mesmer_model._resize_output(labels, seg_input.shape)
        mask   = labels[0, ..., 0].astype(np.int32)

        tifffile.imwrite(mask_path, mask)
        print(f"Processed {img_name}: {mask.max()} cells detected, saved to {mask_path}")

        del nuclear_img, membrane_img, seg_input, labels, mask
        gc.collect()


def segment_imgs_mesmer_onnx(base_dir,
                             input_folder='img_comp',
                             output_folder='masks',
                             markers=['DNA1'],
                             membrane_markers=None,
                             postprocess_kwargs=None,
                             upper_bound=98,
                             lower_bound=2,
                             image_mpp=0.5,
                             overwrite=False,
                             onnx_path=None):
    """
    Segment images using the Mesmer model via ONNX Runtime — no TensorFlow required.

    Works on any platform (Intel Mac, M1/M2, Linux). Requires:
      - mesmer.onnx  (convert once with convert_mesmer_to_onnx.py on an Intel/Linux machine)
      - onnxruntime  (pip install onnxruntime)
      - deepcell-toolbox  (pip install deepcell-toolbox)

    Parameters
    ----------
    onnx_path : str or None
        Path to mesmer.onnx. Defaults to ~/.keras/models/mesmer.onnx.
    postprocess_kwargs : dict or None
        Overrides for the deep_watershed step.
        Example: {'maxima_threshold': 0.075, 'interior_threshold': 0.2}
    """
    import onnxruntime as ort
    from pythonimc._deepcell_toolbox.processing import percentile_threshold, histogram_normalization
    from pythonimc._deepcell_toolbox.utils import tile_image, untile_image, resize
    from pythonimc._deepcell_toolbox.deep_watershed import deep_watershed

    if onnx_path is None:
        # Look in the repo's models/ folder (sibling of the pythonimc/ package)
        onnx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'mesmer.onnx')

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"mesmer.onnx not found at {onnx_path}.\n"
            "Run convert_mesmer_to_onnx.py once on a machine with TensorFlow to generate it."
        )

    print(f"Loading ONNX model from {onnx_path} ...")
    session = ort.InferenceSession(onnx_path)
    model_input_shape = (256, 256)  # fixed by the Mesmer architecture

    img_dir   = os.path.join(base_dir, input_folder)
    masks_dir = os.path.join(base_dir, output_folder)
    os.makedirs(masks_dir, exist_ok=True)

    images = [x for x in io.list_image_files(img_dir)]
    if not images:
        print(f"No images found in {img_dir}.")
        return
    print(f"Found {len(images)} image(s) in {img_dir}")
    random.shuffle(images)

    # Default postprocessing kwargs (same as deepcell's defaults)
    default_pp = {
        'maxima_threshold': 0.075,
        'maxima_smooth': 0,
        'interior_threshold': 0.2,
        'interior_smooth': 2,
        'small_objects_threshold': 15,
        'fill_holes_threshold': 15,
        'radius': 2,
    }
    pp_kwargs = {**default_pp, **(postprocess_kwargs or {})}

    for img_path in images:
        img_name  = os.path.basename(img_path)
        mask_path = os.path.join(masks_dir, img_name)

        if os.path.exists(mask_path) and not overwrite:
            print(f"Mask for {img_name} already exists. Skipping.")
            continue

        nuclear_img = preprocess_img(
            subset_img_channels(base_dir, img_path, markers=markers),
            upper_bound, lower_bound
        )
        if nuclear_img is None:
            print(f"Skipping {img_name}: invalid image.")
            continue

        membrane_img = (
            preprocess_img(
                subset_img_channels(base_dir, img_path, markers=membrane_markers),
                upper_bound, lower_bound
            )
            if membrane_markers is not None
            else np.zeros_like(nuclear_img)
        )

        # Build (1, H, W, 2) input, scale to model_mpp=0.5 if needed
        seg_input = np.stack([nuclear_img, membrane_img], axis=-1)[np.newaxis].astype(np.float32)
        original_shape = seg_input.shape

        if image_mpp not in {None, 0.5}:
            scale = image_mpp / 0.5
            new_hw = (int(original_shape[1] * scale), int(original_shape[2] * scale))
            seg_input = resize(seg_input, new_hw, data_format='channels_last')

        # Mesmer preprocessing (percentile clip + histogram normalisation)
        seg_input = percentile_threshold(seg_input, percentile=99.9)
        seg_input = histogram_normalization(seg_input, kernel_size=128)

        # Tile into 256×256 overlapping patches
        tiles, tiles_info = tile_image(
            seg_input, model_input_shape=model_input_shape, stride_ratio=0.75
        )

        # Run each tile through ONNX
        n_tiles = tiles.shape[0]
        output_tiles = None
        for i in range(n_tiles):
            tile = tiles[i:i+1].astype(np.float32)
            preds = session.run(None, {'input_0': tile})  # list of 4 arrays
            if output_tiles is None:
                output_tiles = [np.zeros((n_tiles,) + p.shape[1:], dtype=np.float32)
                                for p in preds]
            for k, p in enumerate(preds):
                output_tiles[k][i] = p[0]

        # Untile each of the 4 output heads
        output_images = [
            untile_image(output_tiles[k], tiles_info, model_input_shape=model_input_shape)
            for k in range(4)
        ]

        # Format outputs (mirrors format_output_mesmer)
        model_output = {
            'whole-cell': [output_images[0], output_images[1][..., 1:2]],
        }

        # Watershed postprocessing
        label_image = deep_watershed(model_output['whole-cell'], **pp_kwargs)

        # Resize label back to original resolution if we scaled the input
        if image_mpp not in {None, 0.5}:
            label_image = resize(label_image, original_shape[1:3], data_format='channels_last')

        mask = label_image[0, ..., 0].astype(np.int32)

        tifffile.imwrite(mask_path, mask)
        print(f"Processed {img_name}: {mask.max()} cells detected, saved to {mask_path}")

        del nuclear_img, membrane_img, seg_input, tiles, output_tiles, output_images, mask
        gc.collect()


def load_marker_image(img_stack, markers, adjust_contrast, upper_bound, lower_bound,
                      markers_to_channels=None, base_dir=None):
    """
    Given one or more markers, return a single 2D image.
    Uses max projection if multiple markers are passed.
    """
    if isinstance(markers, str):
        markers = [markers]

    if markers_to_channels is None:
        indices = get_channel_indices(base_dir, markers)
    else:
        indices = [markers_to_channels[m] for m in markers]

    imgs = [adjust_contrast(img_stack[idx, :, :], upper_bound, lower_bound).astype(np.float32)
            for idx in indices]

    if len(imgs) > 1:
        return np.max(imgs, axis=0)
    else:
        return imgs[0]
