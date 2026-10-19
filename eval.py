import os
import csv
import cv2
import numpy as np
import torch
import pyiqa


###################################
# DEVICE
###################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_REFERENCE_METRICS = ("psnr", "ssim", "lpips", "ciede2000")
DEFAULT_NO_REFERENCE_METRICS = ("niqe", "nrqm", "brisque", "loe")


###################################
# UTILITIES
###################################
def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images(folder):
    if not os.path.isdir(folder):
        return []

    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and is_image_file(f)
    )


def load_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Backward-compatible alias from your uploaded script.
def load_image(path):
    return load_image_rgb(path)


def to_tensor(img_rgb):
    img = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def build_dict(folder):
    files = list_images(folder)
    return {os.path.splitext(f)[0]: f for f in files}


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(headers, rows):
    table_rows = [[format_value(row.get(h, "")) for h in headers] for row in rows]
    col_widths = []

    for i, header in enumerate(headers):
        max_len = len(header)
        for row in table_rows:
            max_len = max(max_len, len(row[i]))
        col_widths.append(max_len)

    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(separator_line)

    for row in table_rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else None


def safe_std(values):
    return float(np.std(values)) if len(values) > 0 else None


def _tensor_to_rgb_numpy(img_tensor):
    if img_tensor.ndim != 4 or img_tensor.shape[0] != 1 or img_tensor.shape[1] != 3:
        raise ValueError(
            f"Expected tensor shape (1, 3, H, W), got {tuple(img_tensor.shape)}"
        )

    return (
        img_tensor.detach()
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


###################################
# CUSTOM METRICS
###################################
def _delta_e_ciede2000(lab1, lab2):
    """Vectorized CIEDE2000 (Delta E 00) for OpenCV-style CIELAB arrays."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) / 2.0

    c_bar_7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar_7 / (c_bar_7 + 25.0 ** 7)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2

    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime

    delta_h_prime = h2_prime - h1_prime
    zero_chroma = (c1_prime * c2_prime) == 0.0
    delta_h_prime = np.where(zero_chroma, 0.0, delta_h_prime)
    delta_h_prime = np.where(
        (~zero_chroma) & (delta_h_prime > 180.0),
        delta_h_prime - 360.0,
        delta_h_prime,
    )
    delta_h_prime = np.where(
        (~zero_chroma) & (delta_h_prime < -180.0),
        delta_h_prime + 360.0,
        delta_h_prime,
    )

    delta_h_term = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(
        np.radians(delta_h_prime) / 2.0
    )

    l_bar_prime = (l1 + l2) / 2.0
    c_bar_prime = (c1_prime + c2_prime) / 2.0

    h_abs_diff = np.abs(h1_prime - h2_prime)
    h_sum = h1_prime + h2_prime
    h_bar_prime = np.where(
        zero_chroma,
        h_sum,
        np.where(
            h_abs_diff <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )

    delta_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0) ** 2)
    r_c = 2.0 * np.sqrt(
        (c_bar_prime ** 7) / ((c_bar_prime ** 7) + 25.0 ** 7)
    )

    s_l = 1.0 + (
        0.015 * (l_bar_prime - 50.0) ** 2
        / np.sqrt(20.0 + (l_bar_prime - 50.0) ** 2)
    )
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t

    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    delta_e = np.sqrt(
        (delta_l_prime / s_l) ** 2
        + (delta_c_prime / s_c) ** 2
        + (delta_h_term / s_h) ** 2
        + r_t * (delta_c_prime / s_c) * (delta_h_term / s_h)
    )

    return delta_e


class CIEDE2000Metric:
    lower_better = True

    def __call__(self, pred, target):
        pred_rgb = _tensor_to_rgb_numpy(pred)
        target_rgb = _tensor_to_rgb_numpy(target)

        if pred_rgb.shape != target_rgb.shape:
            raise ValueError(
                f"CIEDE2000 shape mismatch: {pred_rgb.shape} vs {target_rgb.shape}"
            )

        pred_lab = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB)
        score = float(np.mean(_delta_e_ciede2000(pred_lab, target_lab)))
        return torch.tensor(score, dtype=torch.float32, device=pred.device)


class LOEMetric:
    lower_better = True

    def __init__(self, min_dimension=50, chunk_size=512):
        self.min_dimension = min_dimension
        self.chunk_size = chunk_size

    def _resize_for_loe(self, img_rgb):
        height, width = img_rgb.shape[:2]
        scale = self.min_dimension / float(min(height, width))

        if scale == 1.0:
            return img_rgb

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(img_rgb, (new_width, new_height), interpolation=interpolation)

    def __call__(self, original, enhanced):
        original_rgb = _tensor_to_rgb_numpy(original)
        enhanced_rgb = _tensor_to_rgb_numpy(enhanced)

        if original_rgb.shape != enhanced_rgb.shape:
            raise ValueError(
                f"LOE shape mismatch: {original_rgb.shape} vs {enhanced_rgb.shape}"
            )

        original_small = self._resize_for_loe(original_rgb)
        enhanced_small = self._resize_for_loe(enhanced_rgb)

        original_lightness = np.max(original_small, axis=2).reshape(-1)
        enhanced_lightness = np.max(enhanced_small, axis=2).reshape(-1)

        num_pixels = original_lightness.size
        total_error = 0

        for start in range(0, num_pixels, self.chunk_size):
            end = min(start + self.chunk_size, num_pixels)

            original_order = (
                original_lightness[start:end, None] >= original_lightness[None, :]
            )
            enhanced_order = (
                enhanced_lightness[start:end, None] >= enhanced_lightness[None, :]
            )

            total_error += np.count_nonzero(np.logical_xor(original_order, enhanced_order))

        score = float(total_error) / float(num_pixels)
        return torch.tensor(score, dtype=torch.float32, device=original.device)


###################################
# METRICS
###################################
def init_metrics(metric_names):
    metrics = {}

    for name in metric_names:
        if name == "ciede2000":
            metrics[name] = CIEDE2000Metric()
        elif name == "loe":
            metrics[name] = LOEMetric()
        else:
            metrics[name] = pyiqa.create_metric(name, device=DEVICE)

    return metrics


def init_reference_metrics(metric_names=DEFAULT_REFERENCE_METRICS):
    return init_metrics(metric_names)


def init_no_reference_metrics(metric_names=DEFAULT_NO_REFERENCE_METRICS):
    return init_metrics(metric_names)


###################################
# CSV
###################################
def save_csv(rows, headers, csv_save_path):
    if csv_save_path is None:
        return

    folder = os.path.dirname(csv_save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(csv_save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {csv_save_path}")


###################################
# REFERENCE EVALUATION
###################################
@torch.no_grad()
def evaluate_reference(
    output_path,
    ground_truth_path,
    csv_save_path=None,
    metric_names=DEFAULT_REFERENCE_METRICS,
):
    print("Using device:", DEVICE)
    print("Running REFERENCE evaluation")
    print("Output path      :", output_path)
    print("Ground-truth path:", ground_truth_path)

    pred_dict = build_dict(output_path)
    gt_dict = build_dict(ground_truth_path)
    common = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

    if len(common) == 0:
        raise RuntimeError("No matching images found between output and ground-truth folders.")

    metrics = init_reference_metrics(metric_names)
    rows = []

    for stem in common:
        pred_name = pred_dict[stem]
        gt_name = gt_dict[stem]
        pred_path = os.path.join(output_path, pred_name)
        gt_path = os.path.join(ground_truth_path, gt_name)

        print("Processing:", pred_name)
        pred = to_tensor(load_image_rgb(pred_path))
        gt = to_tensor(load_image_rgb(gt_path))

        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {pred_name}: {tuple(pred.shape)} vs {tuple(gt.shape)}"
            )

        row = {"image": pred_name}
        for metric_name in metric_names:
            row[metric_name] = float(metrics[metric_name](pred, gt).item())
        rows.append(row)

    mean_row = {"image": "MEAN"}
    std_row = {"image": "STD"}
    for metric_name in metric_names:
        values = [r[metric_name] for r in rows]
        mean_row[metric_name] = safe_mean(values)
        std_row[metric_name] = safe_std(values)

    rows_with_stats = rows + [mean_row, std_row]
    headers = ["image", *metric_names]

    print("\n===== FINAL RESULTS =====")
    print_table(headers, rows_with_stats)

    print("\nMean ± Std Results:")
    for metric_name in metric_names:
        print(
            f"{metric_name.upper():8s}: "
            f"{mean_row[metric_name]:.4f} ± {std_row[metric_name]:.4f}"
        )

    save_csv(rows_with_stats, headers, csv_save_path)
    return rows_with_stats


###################################
# NO-REFERENCE EVALUATION
###################################
@torch.no_grad()
def evaluate_no_reference(
    output_path,
    csv_save_path=None,
    metric_names=DEFAULT_NO_REFERENCE_METRICS,
    input_path=None,
):
    print("Using device:", DEVICE)
    print("Running NO-REFERENCE evaluation")
    print("Output path:", output_path)

    files = list_images(output_path)

    if len(files) == 0:
        raise RuntimeError(f"No images found in output folder: {output_path}")

    use_loe = "loe" in metric_names
    input_dict = None

    if use_loe:
        if not input_path:
            raise ValueError(
                "LOE requires the original low-light images. Provide input_path for "
                "evaluate_no_reference (CLI: --input)."
            )
        if not os.path.isdir(input_path):
            raise RuntimeError(f"LOE input folder not found: {input_path}")

        print("Original input path:", input_path)
        input_dict = build_dict(input_path)

    metrics = init_no_reference_metrics(metric_names)
    rows = []

    for filename in files:
        path = os.path.join(output_path, filename)
        print("Processing:", filename)

        img = to_tensor(load_image_rgb(path))
        row = {"image": filename}

        original = None
        if use_loe:
            stem = os.path.splitext(filename)[0]
            if stem not in input_dict:
                raise RuntimeError(
                    f"No matching original low-light image found for LOE: {filename}"
                )

            original_path = os.path.join(input_path, input_dict[stem])
            original = to_tensor(load_image_rgb(original_path))

            if img.shape != original.shape:
                raise ValueError(
                    f"LOE shape mismatch for {filename}: "
                    f"{tuple(img.shape)} vs {tuple(original.shape)}"
                )

        for metric_name in metric_names:
            if metric_name == "loe":
                row[metric_name] = float(metrics[metric_name](original, img).item())
            else:
                row[metric_name] = float(metrics[metric_name](img).item())

        rows.append(row)

    mean_row = {"image": "MEAN"}
    std_row = {"image": "STD"}
    for metric_name in metric_names:
        values = [r[metric_name] for r in rows]
        mean_row[metric_name] = safe_mean(values)
        std_row[metric_name] = safe_std(values)

    rows_with_stats = rows + [mean_row, std_row]
    headers = ["image", *metric_names]

    print("\n===== FINAL RESULTS =====")
    print_table(headers, rows_with_stats)

    print("\nMean ± Std Results:")
    for metric_name in metric_names:
        print(
            f"{metric_name.upper():8s}: "
            f"{mean_row[metric_name]:.4f} ± {std_row[metric_name]:.4f}"
        )

    save_csv(rows_with_stats, headers, csv_save_path)
    return rows_with_stats
