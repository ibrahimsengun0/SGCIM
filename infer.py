import os
import csv
import torch

from eval import (
    DEVICE,
    DEFAULT_REFERENCE_METRICS,
    DEFAULT_NO_REFERENCE_METRICS,
    init_reference_metrics,
    init_no_reference_metrics,
    list_images,
    load_image_rgb,
    to_tensor,
    safe_mean,
    safe_std,
    print_table,
)


###################################
# DEFAULT CONFIG
###################################
DEFAULT_GT_DATASET_MAP = {
    "lolv1": "lolv1",
    "lol_v1": "lol_v1",
    "lol_v2_real": "lol_v2_real",
    "lolv2_real": "lol_v2_real",
    "lol_v2_syn": "lol_v2_syn",
    "lolv2_syn": "lol_v2_syn",
}

DEFAULT_BATCH_NO_REFERENCE_METRICS = DEFAULT_NO_REFERENCE_METRICS


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
        return {
            "num_images": 0,
            **{name: None for name in metric_names},
            **{f"{name}_std": None for name in metric_names},
        }

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
        **{f"{name}_std": safe_std(metric_values[name]) for name in metric_names},
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
                **{f"{name}_std": format_metric(result[f"{name}_std"]) for name in metric_names},
            }
            rows.append(row)

            metric_text = " | ".join(
                f"{name}={row[name]} ± {row[f'{name}_std']}"
                for name in metric_names
            )
            print(
                f"[REFERENCE] dataset={dataset_name} | method={method_name} | "
                f"images={row['num_images']} | {metric_text}"
            )

        headers = [
            "method",
            "num_images",
            *[item for name in metric_names for item in (name, f"{name}_std")],
        ]
        print(f"\nREFERENCE SUMMARY: {dataset_name}")
        print_table(headers, rows)

        csv_path = os.path.join(save_root, "reference", f"{dataset_name}.csv")
        save_csv(rows, headers, csv_path)


###################################
# NO-REFERENCE EVALUATION
###################################
@torch.no_grad()
def evaluate_no_reference_method(
    method_dir,
    metrics,
    metric_names=DEFAULT_BATCH_NO_REFERENCE_METRICS,
    input_dir=None,
):
    files = list_images(method_dir)

    if len(files) == 0:
        return {
            "num_images": 0,
            **{name: None for name in metric_names},
            **{f"{name}_std": None for name in metric_names},
        }

    use_loe = "loe" in metric_names
    input_dict = None

    if use_loe:
        if not input_dir:
            raise ValueError(
                "LOE requires the original low-light images for each no-reference dataset."
            )
        if not os.path.isdir(input_dir):
            raise RuntimeError(f"LOE input folder not found: {input_dir}")
        input_dict = build_stem_to_file_dict(input_dir)

    metric_values = {name: [] for name in metric_names}

    for filename in files:
        img_path = os.path.join(method_dir, filename)
        img = to_tensor(load_image_rgb(img_path))

        original = None
        if use_loe:
            stem = os.path.splitext(filename)[0]
            if stem not in input_dict:
                raise RuntimeError(
                    f"No matching original low-light image found for LOE: {filename}"
                )

            original_path = os.path.join(input_dir, input_dict[stem])
            original = to_tensor(load_image_rgb(original_path))

            if img.shape != original.shape:
                raise ValueError(
                    f"LOE shape mismatch for {filename}: "
                    f"{tuple(img.shape)} vs {tuple(original.shape)}"
                )

        for metric_name in metric_names:
            if metric_name == "loe":
                metric_values[metric_name].append(
                    metrics[metric_name](original, img).item()
                )
            else:
                metric_values[metric_name].append(metrics[metric_name](img).item())

    return {
        "num_images": len(files),
        **{name: safe_mean(metric_values[name]) for name in metric_names},
        **{f"{name}_std": safe_std(metric_values[name]) for name in metric_names},
    }


def evaluate_no_reference_root(
    no_reference_root,
    save_root,
    metric_names=DEFAULT_BATCH_NO_REFERENCE_METRICS,
    input_root="",
):
    if not os.path.isdir(no_reference_root):
        print(f"[Skip] No-reference root not found: {no_reference_root}")
        return

    print(f"\nUsing device: {DEVICE}")
    print("Running NO-REFERENCE batch evaluation")

    if "loe" in metric_names and not input_root:
        raise ValueError(
            "LOE requires the original low-light dataset root. "
            "Provide input_root (CLI: --input-root), organized as "
            "input_root/dataset_name/images."
        )

    metrics = init_no_reference_metrics(metric_names)
    dataset_names = list_subdirs(no_reference_root)

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(no_reference_root, dataset_name)
        input_dir = os.path.join(input_root, dataset_name) if input_root else None
        method_names = list_subdirs(dataset_dir)

        if len(method_names) == 0:
            print(f"[Skip] No method folders in: {dataset_dir}")
            continue

        if "loe" in metric_names and not os.path.isdir(input_dir):
            raise RuntimeError(
                f"LOE input folder not found for dataset '{dataset_name}': {input_dir}"
            )

        rows = []

        for method_name in method_names:
            method_dir = os.path.join(dataset_dir, method_name)
            result = evaluate_no_reference_method(
                method_dir,
                metrics,
                metric_names,
                input_dir=input_dir,
            )

            row = {
                "method": method_name,
                "num_images": result["num_images"],
                **{name: format_metric(result[name]) for name in metric_names},
                **{f"{name}_std": format_metric(result[f"{name}_std"]) for name in metric_names},
            }
            rows.append(row)

            metric_text = " | ".join(
                f"{name}={row[name]} ± {row[f'{name}_std']}"
                for name in metric_names
            )
            print(
                f"[NO-REFERENCE] dataset={dataset_name} | method={method_name} | "
                f"images={row['num_images']} | {metric_text}"
            )

        headers = [
            "method",
            "num_images",
            *[item for name in metric_names for item in (name, f"{name}_std")],
        ]
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
    input_root="",
):
    """
    Run multi-dataset evaluation for reference and/or no-reference LLIE results.

    Expected result folder structure:

    results_root/
        reference/
            dataset_name/
                method_name/
                    image files
        no_reference/
            dataset_name/
                method_name/
                    image files

    For LOE only, original low-light images are read from:

    input_root/
        dataset_name/
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
    print(f"Input root   : {input_root if input_root else '(not provided)'}")
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
            input_root=input_root,
        )

    print("\nDone.")
