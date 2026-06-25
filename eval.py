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
DEFAULT_REFERENCE_METRICS = ("psnr", "ssim", "lpips")
DEFAULT_NO_REFERENCE_METRICS = ("niqe", "nrqm", "brisque")


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


###################################
# METRICS
###################################
def init_metrics(metric_names):
    return {name: pyiqa.create_metric(name, device=DEVICE) for name in metric_names}


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
    for metric_name in metric_names:
        mean_row[metric_name] = safe_mean([r[metric_name] for r in rows])

    rows_with_mean = rows + [mean_row]
    headers = ["image", *metric_names]

    print("\n===== FINAL RESULTS =====")
    print_table(headers, rows_with_mean)

    print("\nMean Results:")
    for metric_name in metric_names:
        print(f"{metric_name.upper():8s}: {mean_row[metric_name]:.4f}")

    save_csv(rows_with_mean, headers, csv_save_path)
    return rows_with_mean


###################################
# NO-REFERENCE EVALUATION
###################################
@torch.no_grad()
def evaluate_no_reference(
    output_path,
    csv_save_path=None,
    metric_names=DEFAULT_NO_REFERENCE_METRICS,
):
    print("Using device:", DEVICE)
    print("Running NO-REFERENCE evaluation")
    print("Output path:", output_path)

    files = list_images(output_path)

    if len(files) == 0:
        raise RuntimeError(f"No images found in output folder: {output_path}")

    metrics = init_no_reference_metrics(metric_names)
    rows = []

    for filename in files:
        path = os.path.join(output_path, filename)
        print("Processing:", filename)

        img = to_tensor(load_image_rgb(path))
        row = {"image": filename}

        for metric_name in metric_names:
            row[metric_name] = float(metrics[metric_name](img).item())

        rows.append(row)

    mean_row = {"image": "MEAN"}
    for metric_name in metric_names:
        mean_row[metric_name] = safe_mean([r[metric_name] for r in rows])

    rows_with_mean = rows + [mean_row]
    headers = ["image", *metric_names]

    print("\n===== FINAL RESULTS =====")
    print_table(headers, rows_with_mean)

    print("\nMean Results:")
    for metric_name in metric_names:
        print(f"{metric_name.upper():8s}: {mean_row[metric_name]:.4f}")

    save_csv(rows_with_mean, headers, csv_save_path)
    return rows_with_mean
