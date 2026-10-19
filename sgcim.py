import os
import cv2
import numpy as np


###################################
# IMAGE I/O
###################################
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images(folder_path, recursive=False):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    image_paths = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for name in files:
                if is_image_file(name):
                    image_paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            if os.path.isfile(path) and is_image_file(name):
                image_paths.append(path)

    return sorted(image_paths)


def load_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images(folder_path, recursive=False):
    images = []
    for image_path in list_images(folder_path, recursive=recursive):
        rel_name = os.path.relpath(image_path, folder_path)
        images.append((rel_name, load_image_rgb(image_path)))
    return images


def save_image_rgb(image, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr)
    if not ok:
        raise ValueError(f"Failed to save image: {path}")


def save_image(image, path):
    save_image_rgb(image, path)


###################################
# IMAGE STATISTICS (NEW METHOD)
###################################
def compute_entropy(v):
    hist, _ = np.histogram(v, bins=256, range=(0, 1))
    p = hist.astype(np.float64)

    s = p.sum()
    if s < 1e-8:
        return 0.0

    p = p / s
    entropy = -np.sum(p * np.log2(p + 1e-12))
    max_entropy = np.log2(256)

    return float(entropy / max_entropy)


def compute_contrast(v):
    v64 = np.ascontiguousarray(v.astype(np.float64))
    lap = cv2.Laplacian(v64, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))


def compute_score(v):
    v = np.ascontiguousarray(v.astype(np.float32))

    brightness = float(np.mean(v))
    entropy = compute_entropy(v)
    contrast = compute_contrast(v)

    contrast = np.clip(contrast / 0.1, 0.0, 1.0)

    score = (
        0.5 * (1.0 - brightness) +
        0.2 * (1.0 - entropy) +
        0.3 * (1.0 - contrast)
    )

    return float(np.clip(score, 0.0, 1.0))


###################################
# ENHANCEMENT (NEW LOGIC)
###################################
def gamma_correction(x, g):
    return np.power(
        np.clip(x, 0, 1).astype(np.float32),
        np.float32(g)
    ).astype(np.float32)


def percentile_stretch(x, low_pct=1.0, high_pct=99.0):
    # kept signature for compatibility (old pipeline)
    p1 = np.percentile(x, 1)
    p2 = np.percentile(x, 99)

    if p2 - p1 < 1e-6:
        return x.astype(np.float32)

    return np.clip(
        (x - p1) / (p2 - p1),
        0,
        1
    ).astype(np.float32)


def enhance_image(image, low_pct=1.0, high_pct=99.0):
    img = image.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)

    # NEW SCORE
    score = compute_score(v)

    # gamma mapping (new)
    gamma_val = np.exp(-score)

    # contrast/stretch
    v = percentile_stretch(v)
    v = gamma_correction(v, gamma_val)

    # saturation scaling (new)
    s_scale = np.float32(np.exp(0.5 - score))
    s = np.clip(s * s_scale, 0, 1).astype(np.float32)

    hsv_out = np.stack([h, s, v], axis=2).astype(np.float32)

    rgb = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)


###################################
# DATASET PROCESSING (UNCHANGED)
###################################
def process_dataset(images, output_folder, low_pct=1.0, high_pct=99.0):
    os.makedirs(output_folder, exist_ok=True)

    for image_name, image in images:
        enhanced = enhance_image(image, low_pct=low_pct, high_pct=high_pct)
        save_path = os.path.join(output_folder, image_name)
        save_image_rgb(enhanced, save_path)


def enhance_folder(input_folder, output_folder, recursive=False, low_pct=1.0, high_pct=99.0):
    image_paths = list_images(input_folder, recursive=recursive)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        rel_name = os.path.relpath(image_path, input_folder)
        save_path = os.path.join(output_folder, rel_name)

        print(f"Enhancing: {rel_name}")
        image = load_image_rgb(image_path)
        enhanced = enhance_image(image, low_pct=low_pct, high_pct=high_pct)
        save_image_rgb(enhanced, save_path)

    print(f"Saved enhanced images to: {output_folder}")
    return output_folder
