import argparse

from sgcim import enhance_folder
from eval import (
    DEFAULT_REFERENCE_METRICS,
    DEFAULT_NO_REFERENCE_METRICS,
    evaluate_reference,
    evaluate_no_reference,
)
from infer import (
    DEFAULT_BATCH_NO_REFERENCE_METRICS,
    DEFAULT_GT_DATASET_MAP,
    multiple_dataset_infer,
)


EXAMPLES = """
Examples:
  # Enhance one folder
  python main.py enhance --input data/low --output results/edlr

  # Enhance recursively
  python main.py enhance --input data/low --output results/edlr --recursive

  # Evaluate enhanced images against ground truth
  python main.py eval-single --output results/edlr --gt data/high --csv eval_csv/edlr_ref.csv

  # Evaluate enhanced images without ground truth
  python main.py eval-single --output results/edlr --mode no_reference --csv eval_csv/edlr_nr.csv

  # Enhance then evaluate in one command
  python main.py pipeline --input data/low --output results/edlr --gt data/high --csv eval_csv/edlr_ref.csv

  # Batch evaluation layout:
  # results_root/reference/dataset_name/method_name/images
  # results_root/no_reference/dataset_name/method_name/images
  python main.py eval-batch --results-root results_root --gt-root gt_root --save-root eval_csv --mode all

  # Batch evaluation with custom GT mapping
  python main.py eval-batch --results-root results_root --gt-root gt_root --map my_dataset=lolv1
"""


def parse_metric_list(values, default):
    if values is None or len(values) == 0:
        return tuple(default)
    return tuple(values)


def parse_gt_dataset_map(pairs):
    gt_map = dict(DEFAULT_GT_DATASET_MAP)

    if pairs is None:
        return gt_map

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --map value '{pair}'. Expected format: result_dataset=gt_dataset")

        result_dataset, gt_dataset = pair.split("=", 1)
        result_dataset = result_dataset.strip()
        gt_dataset = gt_dataset.strip()

        if not result_dataset or not gt_dataset:
            raise ValueError(f"Invalid --map value '{pair}'. Expected format: result_dataset=gt_dataset")

        gt_map[result_dataset] = gt_dataset

    return gt_map


def normalize_mode(mode):
    return mode.replace("-", "_")


def run_enhance(args):
    enhance_folder(
        input_folder=args.input,
        output_folder=args.output,
        recursive=args.recursive,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
    )


def run_eval_single(args):
    mode = normalize_mode(args.mode)

    if mode == "auto":
        mode = "reference" if args.gt else "no_reference"

    if mode == "reference":
        if not args.gt:
            raise ValueError("--gt is required when --mode reference is used.")

        evaluate_reference(
            output_path=args.output,
            ground_truth_path=args.gt,
            csv_save_path=args.csv,
            metric_names=parse_metric_list(args.ref_metrics, DEFAULT_REFERENCE_METRICS),
        )
    elif mode == "no_reference":
        evaluate_no_reference(
            output_path=args.output,
            csv_save_path=args.csv,
            metric_names=parse_metric_list(args.nr_metrics, DEFAULT_NO_REFERENCE_METRICS),
        )
    else:
        raise ValueError("Invalid eval-single mode. Use auto, reference, or no_reference.")


def run_eval_batch(args):
    multiple_dataset_infer(
        results_root=args.results_root,
        gt_root=args.gt_root,
        save_root=args.save_root,
        mode=normalize_mode(args.mode),
        gt_dataset_map=parse_gt_dataset_map(args.map),
        reference_metrics=parse_metric_list(args.ref_metrics, DEFAULT_REFERENCE_METRICS),
        no_reference_metrics=parse_metric_list(args.nr_metrics, DEFAULT_BATCH_NO_REFERENCE_METRICS),
    )


def run_pipeline(args):
    enhance_folder(
        input_folder=args.input,
        output_folder=args.output,
        recursive=args.recursive,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
    )

    mode = normalize_mode(args.eval_mode)
    if mode == "auto":
        mode = "reference" if args.gt else "no_reference"

    if mode == "reference":
        if not args.gt:
            raise ValueError("--gt is required for reference evaluation in pipeline mode.")

        evaluate_reference(
            output_path=args.output,
            ground_truth_path=args.gt,
            csv_save_path=args.csv,
            metric_names=parse_metric_list(args.ref_metrics, DEFAULT_REFERENCE_METRICS),
        )
    elif mode == "no_reference":
        evaluate_no_reference(
            output_path=args.output,
            csv_save_path=args.csv,
            metric_names=parse_metric_list(args.nr_metrics, DEFAULT_NO_REFERENCE_METRICS),
        )
    elif mode == "none":
        print("Pipeline enhancement finished. Evaluation skipped.")
    else:
        raise ValueError("Invalid pipeline --eval-mode. Use auto, reference, no_reference, or none.")


def add_common_enhance_args(parser):
    parser.add_argument("--input", required=True, help="Input folder containing low-light images.")
    parser.add_argument("--output", required=True, help="Output folder for enhanced images.")
    parser.add_argument("--recursive", action="store_true", help="Process images inside subfolders too.")
    parser.add_argument("--low-pct", type=float, default=1.0, help="Low percentile for V-channel stretch.")
    parser.add_argument("--high-pct", type=float, default=99.0, help="High percentile for V-channel stretch.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="EDLR enhancement and evaluation command-line interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    enhance_parser = subparsers.add_parser("enhance", help="Enhance a folder of images with EDLR.")
    add_common_enhance_args(enhance_parser)
    enhance_parser.set_defaults(func=run_enhance)

    eval_single_parser = subparsers.add_parser("eval-single", help="Evaluate one output folder.")
    eval_single_parser.add_argument("--output", required=True, help="Folder containing enhanced/output images.")
    eval_single_parser.add_argument("--gt", default="", help="Ground-truth folder for reference metrics.")
    eval_single_parser.add_argument("--csv", default=None, help="Optional CSV save path.")
    eval_single_parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "reference", "no_reference", "no-reference"],
        help="Evaluation mode. auto uses reference if --gt is provided, otherwise no_reference.",
    )
    eval_single_parser.add_argument(
        "--ref-metrics",
        nargs="*",
        default=None,
        help="Reference metrics to compute. Default: psnr ssim lpips",
    )
    eval_single_parser.add_argument(
        "--nr-metrics",
        nargs="*",
        default=None,
        help="No-reference metrics to compute. Default: niqe nrqm brisque musiq",
    )
    eval_single_parser.set_defaults(func=run_eval_single)

    eval_batch_parser = subparsers.add_parser("eval-batch", help="Evaluate multiple datasets/methods.")
    eval_batch_parser.add_argument("--results-root", required=True, help="Root folder containing reference/ and/or no_reference/.")
    eval_batch_parser.add_argument("--gt-root", default="", help="GT root folder for reference datasets.")
    eval_batch_parser.add_argument("--save-root", default="eval_csv", help="Folder where CSV summaries are saved.")
    eval_batch_parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "reference", "no_reference", "no-reference"],
        help="Batch evaluation mode.",
    )
    eval_batch_parser.add_argument(
        "--map",
        action="append",
        default=None,
        help="Dataset mapping result_dataset=gt_dataset. Can be used multiple times.",
    )
    eval_batch_parser.add_argument(
        "--ref-metrics",
        nargs="*",
        default=None,
        help="Reference metrics to compute. Default: psnr ssim lpips",
    )
    eval_batch_parser.add_argument(
        "--nr-metrics",
        nargs="*",
        default=None,
        help="No-reference metrics to compute. Default: niqe nrqm brisque",
    )
    eval_batch_parser.set_defaults(func=run_eval_batch)

    pipeline_parser = subparsers.add_parser("pipeline", help="Enhance images, then optionally evaluate them.")
    add_common_enhance_args(pipeline_parser)
    pipeline_parser.add_argument("--gt", default="", help="Ground-truth folder for reference evaluation.")
    pipeline_parser.add_argument("--csv", default=None, help="Optional CSV save path.")
    pipeline_parser.add_argument(
        "--eval-mode",
        default="auto",
        choices=["auto", "reference", "no_reference", "no-reference", "none"],
        help="Evaluation mode after enhancement.",
    )
    pipeline_parser.add_argument(
        "--ref-metrics",
        nargs="*",
        default=None,
        help="Reference metrics to compute. Default: psnr ssim lpips",
    )
    pipeline_parser.add_argument(
        "--nr-metrics",
        nargs="*",
        default=None,
        help="No-reference metrics to compute. Default: niqe nrqm brisque musiq",
    )
    pipeline_parser.set_defaults(func=run_pipeline)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
