import os
import csv
import torch

from eval import (
    DEVICE,
    DEFAULT_REFERENCE_METRICS,
    init_reference_metrics,
    init_no_reference_metrics,
    list_images,
    load_image_rgb,
    to_tensor,
    safe_mean,
    print_table,
)


###################################
# DEFAULT CONFIG
###################################
DEFAULT_GT_DATASET_MAP = {
    "lolv1": "lolv1",
    "lol_v1": "lolv1",
    "lol_v2_real": "lol_v2_real",
    "lolv2_real": "lol_v2_real",
    "lol_v2_syn": "lol_v2_syn",
    "lolv2_syn": "lol_v2_syn",
}

DEFAULT_BATCH_NO_REFERENCE_METRICS = ("niqe", "nrqm", "brisque")


###################################
# UTILITIES
###################################
def build_stem_to_file_dict(folder):
    files = list_images(folder)
    return {os.path.splitext(f)[0]: f for f in files}


def format_metric(x):
    if x is None:
        return ""
    return f"{x:.6f}"


def save_csv(rows, headers, csv_path):
    folder = os.path.dirname(csv_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {csv_path}")


def list_subdirs(folder):
    if not os.path.isdir(folder):
        return []

    return sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    )


def resolve_gt_dir(dataset_name, gt_root, gt_dataset_map):
    mapped_name = gt_dataset_map.get(dataset_name, dataset_name)
    return os.path.join(gt_root, mapped_name)


###################################
# REFERENCE EVALUATION
###################################
@torch.no_grad()
def evaluate_reference_method(method_dir, gt_dir, metrics, metric_names=DEFAULT_REFERENCE_METRICS):
    pred_dict = build_stem_to_file_dict(method_dir)
    gt_dict = build_stem_to_file_dict(gt_dir)
    common_keys = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))

    if len(common_keys) == 0:
        return {"num_images": 0, **{name: None for name in metric_names}}

    metric_values = {name: [] for name in metric_names}

    for stem in common_keys:
        pred_path = os.path.join(method_dir, pred_dict[stem])
        gt_path = os.path.join(gt_dir, gt_dict[stem])

        pred = to_tensor(load_image_rgb(pred_path))
        gt = to_tensor(load_image_rgb(gt_path))

        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {pred_dict[stem]}: {tuple(pred.shape)} vs {tuple(gt.shape)}"
            )

        for metric_name in metric_names:
            metric_values[metric_name].append(metrics[metric_name](pred, gt).item())

    return {
        "num_images": len(common_keys),
        **{name: safe_mean(metric_values[name]) for name in metric_names},
    }


def evaluate_reference_root(
    reference_root,
    gt_root,
    save_root,
    gt_dataset_map,
    metric_names=DEFAULT_REFERENCE_METRICS,
):
    if not os.path.isdir(reference_root):
        print(f"[Skip] Reference root not found: {reference_root}")
        return

    print(f"\nUsing device: {DEVICE}")
    print("Running REFERENCE batch evaluation")

    metrics = init_reference_metrics(metric_names)
    dataset_names = list_subdirs(reference_root)

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(reference_root, dataset_name)
        gt_dir = resolve_gt_dir(dataset_name, gt_root, gt_dataset_map)

        if not os.path.isdir(gt_dir):
            print(f"[Skip] GT folder not found for dataset '{dataset_name}': {gt_dir}")
            continue

        method_names = list_subdirs(dataset_dir)

        if len(method_names) == 0:
            print(f"[Skip] No method folders in: {dataset_dir}")
            continue

        rows = []

        for method_name in method_names:
            method_dir = os.path.join(dataset_dir, method_name)
            result = evaluate_reference_method(method_dir, gt_dir, metrics, metric_names)

            row = {
                "method": method_name,
                "num_images": result["num_images"],
                **{name: format_metric(result[name]) for name in metric_names},
            }
            rows.append(row)

            metric_text = " | ".join(f"{name}={row[name]}" for name in metric_names)
            print(
                f"[REFERENCE] dataset={dataset_name} | method={method_name} | "
                f"images={row['num_images']} | {metric_text}"
            )

        headers = ["method", "num_images", *metric_names]
        print(f"\nREFERENCE SUMMARY: {dataset_name}")
        print_table(headers, rows)

        csv_path = os.path.join(save_root, "reference", f"{dataset_name}.csv")
        save_csv(rows, headers, csv_path)


###################################
# NO-REFERENCE EVALUATION
###################################
@torch.no_grad()
def evaluate_no_reference_method(method_dir, metrics, metric_names=DEFAULT_BATCH_NO_REFERENCE_METRICS):
    files = list_images(method_dir)

    if len(files) == 0:
        return {"num_images": 0, **{name: None for name in metric_names}}

    metric_values = {name: [] for name in metric_names}

    for filename in files:
        img_path = os.path.join(method_dir, filename)
        img = to_tensor(load_image_rgb(img_path))

        for metric_name in metric_names:
            metric_values[metric_name].append(metrics[metric_name](img).item())

    return {
        "num_images": len(files),
        **{name: safe_mean(metric_values[name]) for name in metric_names},
    }


def evaluate_no_reference_root(
    no_reference_root,
    save_root,
    metric_names=DEFAULT_BATCH_NO_REFERENCE_METRICS,
):
    if not os.path.isdir(no_reference_root):
        print(f"[Skip] No-reference root not found: {no_reference_root}")
        return

    print(f"\nUsing device: {DEVICE}")
    print("Running NO-REFERENCE batch evaluation")

    metrics = init_no_reference_metrics(metric_names)
    dataset_names = list_subdirs(no_reference_root)

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(no_reference_root, dataset_name)
        method_names = list_subdirs(dataset_dir)

        if len(method_names) == 0:
            print(f"[Skip] No method folders in: {dataset_dir}")
            continue

        rows = []

        for method_name in method_names:
            method_dir = os.path.join(dataset_dir, method_name)
            result = evaluate_no_reference_method(method_dir, metrics, metric_names)

            row = {
                "method": method_name,
                "num_images": result["num_images"],
                **{name: format_metric(result[name]) for name in metric_names},
            }
            rows.append(row)

            metric_text = " | ".join(f"{name}={row[name]}" for name in metric_names)
            print(
                f"[NO-REFERENCE] dataset={dataset_name} | method={method_name} | "
                f"images={row['num_images']} | {metric_text}"
            )

        headers = ["method", "num_images", *metric_names]
        print(f"\nNO-REFERENCE SUMMARY: {dataset_name}")
        print_table(headers, rows)

        csv_path = os.path.join(save_root, "no_reference", f"{dataset_name}.csv")
        save_csv(rows, headers, csv_path)


###################################
# SINGLE ENTRY FUNCTION
###################################
def multiple_dataset_infer(
    results_root,
    gt_root="",
    save_root="eval_csv",
    mode="all",
    gt_dataset_map=None,
    reference_metrics=DEFAULT_REFERENCE_METRICS,
    no_reference_metrics=DEFAULT_BATCH_NO_REFERENCE_METRICS,
):
    """
    Run multi-dataset evaluation for reference and/or no-reference LLIE results.

    Expected folder structure:

    results_root/
        reference/
            dataset_name/
                method_name/
                    image files
        no_reference/
            dataset_name/
                method_name/
                    image files
    """
    mode = mode.replace("-", "_")

    if mode not in ["all", "reference", "no_reference"]:
        raise ValueError(
            f"Invalid mode: {mode}. Expected one of: 'all', 'reference', 'no_reference'."
        )

    if gt_dataset_map is None:
        gt_dataset_map = DEFAULT_GT_DATASET_MAP

    reference_root = os.path.join(results_root, "reference")
    no_reference_root = os.path.join(results_root, "no_reference")

    print("=" * 80)
    print("Batch evaluation started")
    print(f"Device       : {DEVICE}")
    print(f"Results root : {results_root}")
    print(f"GT root      : {gt_root if gt_root else '(not provided)'}")
    print(f"Save root    : {save_root}")
    print(f"Mode         : {mode}")
    print("=" * 80)

    if mode in ["all", "reference"]:
        if not gt_root:
            print("[Skip] Reference evaluation skipped because gt_root was not provided.")
        else:
            evaluate_reference_root(
                reference_root=reference_root,
                gt_root=gt_root,
                save_root=save_root,
                gt_dataset_map=gt_dataset_map,
                metric_names=reference_metrics,
            )

    if mode in ["all", "no_reference"]:
        evaluate_no_reference_root(
            no_reference_root=no_reference_root,
            save_root=save_root,
            metric_names=no_reference_metrics,
        )

    print("\nDone.")
