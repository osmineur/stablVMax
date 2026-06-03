# pixelclustering.py

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from minisom import MiniSom
from skimage.morphology import disk, dilation, erosion, remove_small_holes, remove_small_objects
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os
import joblib
import pickle

# ---------------- Model Save/Load ----------------
def save_model(base_dir, som, kmeans, scaler, folder="pixel_model"):
    """Save SOM, KMeans, and Scaler to a directory."""
    out_dir = os.path.join(base_dir, folder)
    os.makedirs(out_dir, exist_ok=True)

    # Save scaler and kmeans with joblib
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(out_dir, "kmeans.pkl"))

    # Save SOM with pickle
    with open(os.path.join(out_dir, "som.pkl"), "wb") as f:
        pickle.dump(som, f)

    print(f"✅ Model saved to {out_dir}")


def load_model(base_dir, folder="pixel_model"):
    """Load SOM, KMeans, and Scaler from a directory."""
    in_dir = os.path.join(base_dir, folder)
    scaler = joblib.load(os.path.join(in_dir, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(in_dir, "kmeans.pkl"))

    with open(os.path.join(in_dir, "som.pkl"), "rb") as f:
        som = pickle.load(f)

    print(f"✅ Model loaded from {in_dir}")
    return som, kmeans, scaler


# ---------------- Preprocessing ----------------
def preprocess_image(img, sigma=4):
    """Gaussian blur + clip per channel at percentile."""
    #img = np.clip(img, 0,1)
    img_blur = np.stack([gaussian_filter(ch, sigma=sigma) for ch in img], axis=0)
    return img_blur

def preprocess_image_long(img, sigma=4, threshold=0.1):
    processed_channels = []
    for ch in img:
        # 1. Binarize
        ch_bin = (ch > threshold).astype(np.uint8)
        ch_filled = remove_small_holes(ch_bin, area_threshold=100, connectivity=2)
        ch_filled = dilation(ch_filled, disk(2))
        ch_eroded = erosion(ch_filled, disk(2))
        ch_blur = gaussian_filter(ch_eroded.astype(float), sigma=sigma)
        processed_channels.append(ch_blur)
    return np.stack(processed_channels, axis=0)


# ---------------- SOM + meta clustering ----------------
def train_som_meta(X, n_clusters=5, som_grid=(5,5), sigma_som=1.0, lr=0.5, n_iter=50000):
    som = MiniSom(som_grid[0], som_grid[1], X.shape[1],
                  sigma=sigma_som, learning_rate=lr, random_seed=42)
    som.train_batch(X, n_iter)
    weights = som.get_weights().reshape(-1, X.shape[1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(weights)
    return som, kmeans

def assign_clusters(X, som, kmeans, scaler):
    X_scaled = scaler.transform(X)
    som_nodes = np.array([som.winner(x)[0]*som.get_weights().shape[1] + som.winner(x)[1] 
                          for x in X_scaled])
    return kmeans.labels_[som_nodes]

# ---------------- Sampling ----------------
def sample_pixels(base_dir, img_folder, channels, n_total=1_000_000, sigma=3):
    """Load images and randomly sample pixels across them."""
    image_path = os.path.join(base_dir, img_folder)
    images = [f for f in os.listdir(image_path) if f.endswith(".tiff")]
    sampled_pixels = []
    n_per_image = n_total // len(images)

    for i in tqdm(images):
        img = tifffile.imread(os.path.join(image_path, i))
        img = img[channels, :, :]   # subset channels
        img = preprocess_image_long(img, sigma=sigma)
        X = np.arcsinh(img.reshape(img.shape[0], -1).T.astype(np.float32))

        # sample subset
        n_pixels = X.shape[0]
        n_sample = min(n_pixels, n_per_image)
        idx = np.random.choice(n_pixels, n_sample, replace=False)
        sampled_pixels.append(X[idx])
        del i 

    return np.vstack(sampled_pixels)

# ---------------- Main training function ----------------
def pixel_training(
    base_dir,
    img_folder,
    channels,
    X_train = None,
    n_train_total=5_000_000,
    n_clusters=5,
    som_grid=(5,5),
    sigma=4,
    write_model=False,
    model_name="pixel_model.npz"
):
    """Train SOM + meta clustering on sampled pixels."""
    # sample training pixels
    if X_train is None:
        X_train = sample_pixels(base_dir, img_folder, channels, n_total=n_train_total, sigma=sigma)
    print("Training pixels:", X_train.shape)

    # scale + train
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    som, kmeans = train_som_meta(X_scaled, n_clusters=n_clusters, som_grid=som_grid)

    if write_model:
        model_path = os.path.join(base_dir, model_name)
        save_model(base_dir, som, kmeans, scaler, folder="pixel_model")
        print(f"Model saved to {model_path}")

    return X_train, som, kmeans, scaler

# ---------------- Visualization ----------------
def reconstruct_cluster_image(labels, shape_hw):
    return labels.reshape(shape_hw)

def plot_clustered_image(cluster_img, cmap="tab20"):
    plt.figure(figsize=(12, 12))
    plt.imshow(cluster_img, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Cluster ID")
    plt.axis("off")
    plt.show()


def clean_cluster_mask(cluster_img, cluster_id, min_size=200, hole_size=1000):
    """Return cleaned binary mask for a given cluster."""
    mask = cluster_img == cluster_id
    mask = remove_small_holes(mask, area_threshold=hole_size)
    mask = remove_small_objects(mask, min_size=min_size)
    return mask


def extract_masks(cluster_img, cluster_map, min_size=200, hole_size=1000):
    masks = {}
    for cid, name in cluster_map.items():
        masks[name] = clean_cluster_mask(cluster_img, cid, min_size=min_size, hole_size=hole_size)
    return masks

def create_compartment_mask_from_clusters(stroma_mask, tumor_mask, muscle_mask, background_mask, selem_radius=20):
    """Create compartments from FlowSOM-derived stroma and tumor masks."""
    selem = disk(selem_radius)

    muscle_zone = erosion(muscle_mask, selem)
    tumor_core = erosion(tumor_mask, selem)
    tumor_border = tumor_mask & ~tumor_core
    stroma_core = ~dilation(tumor_mask, selem) & ~muscle_zone
    stroma_border = stroma_mask & ~tumor_mask & ~stroma_core & ~muscle_zone
    

    # Initialize
    final_mask = np.zeros_like(tumor_mask, dtype=np.uint8)

    zones = [
        (stroma_core, 1),     # stroma core
        (stroma_border, 2),   # stroma border
        (tumor_border, 3),    # tumor border
        (tumor_core, 4),      # tumor core
        (muscle_zone, 5)
    ]
    for zone_mask, label in zones:
        final_mask[zone_mask] = label
    
    final_mask[background_mask] = 0
    return final_mask

def cluster_single_image(base_dir, img_folder, img_name, channels, model_folder, cluster_map, som=None, kmeans=None, scaler=None, sigma=4,  show=True):
    img = tifffile.imread(os.path.join(base_dir, img_folder, img_name))
    img = img[channels, :, :]
    H, W = img.shape[1], img.shape[2]
    img_pre = preprocess_image_long(img, sigma=sigma)
    X_img = np.arcsinh(img_pre.reshape(img.shape[0], -1).T.astype(np.float32))
    if som is None:
        som, kmeans, scaler = load_model(base_dir, folder=model_folder)
    labels_img = assign_clusters(X_img, som, kmeans, scaler)
    cluster_img = reconstruct_cluster_image(labels_img, (H, W))
    cluster_mask = cluster_img.astype(np.uint8)

    if show:
        plot_clustered_image(cluster_mask)
    
    # Clean and extract
    masks = extract_masks(cluster_mask, cluster_map, min_size=200, hole_size=1000)

    # Get stroma/tumor masks
    stroma_mask = masks["Stroma"]
    tumor_mask = masks["Tumor"]
    background_mask = masks["Background"]
    muscle_mask = masks["Muscle"]

    # Make compartments
    final_mask = create_compartment_mask_from_clusters(stroma_mask, tumor_mask, muscle_mask, background_mask, selem_radius=20)

    tiff_path = os.path.join(base_dir, "tmp", f"{img_name}_pixelmask.tiff")
    tifffile.imsave(tiff_path, final_mask)
    print(f"[✓] Saved TIFF to {tiff_path}")

    if show:
        plt.imshow(final_mask, cmap="nipy_spectral")
        plt.colorbar()
        plt.title("Compartment Mask")
        plt.show()