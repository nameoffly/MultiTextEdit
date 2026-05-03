#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import CATEGORY_SIZES, build_output_data, scan_dataset
else:
    from .common import CATEGORY_SIZES, build_output_data, scan_dataset


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **_kwargs):
        return iterable


METRIC_NAMES = ["mse", "psnr"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def require_cv2_numpy():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "evaluate_mse_psnr_masked.py requires opencv-python, numpy, and tqdm to run."
        ) from exc
    return cv2, np


def combine_masks(mask_paths: List[Path]):
    cv2, _ = require_cv2_numpy()

    combined_mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
    if combined_mask is None:
        return None

    for path in mask_paths[1:]:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape != combined_mask.shape:
            mask = cv2.resize(
                mask,
                (combined_mask.shape[1], combined_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask


def align_images(original, edited):
    cv2, np = require_cv2_numpy()

    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_edited = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_original, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_edited, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50),
    )

    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error:
        return None

    if not matches or len(matches[0]) != 2:
        return None

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches) < 4:
        return None

    src_pts = np.float32(
        [keypoints1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
    if matrix is None:
        return None

    height, width = original.shape[:2]
    return cv2.warpAffine(
        edited,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def calculate_masked_metrics(original, aligned, mask) -> Tuple[Optional[float], Optional[float]]:
    cv2, np = require_cv2_numpy()

    if aligned.shape != original.shape:
        aligned = cv2.resize(aligned, (original.shape[1], original.shape[0]))

    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(
            mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    inv_mask = (mask < 128).astype(np.uint8)
    background_pixel_count = np.sum(inv_mask)
    if background_pixel_count == 0:
        return None, None

    original_float = original.astype(np.float32)
    aligned_float = aligned.astype(np.float32)
    squared_error = (original_float - aligned_float) ** 2
    masked_squared_error = squared_error * inv_mask[..., None]

    total_data_points = background_pixel_count * 3
    mse = float(np.sum(masked_squared_error) / total_data_points)
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))
    return mse, psnr


def process_sample(sample: Dict) -> Optional[Dict]:
    cv2, _ = require_cv2_numpy()

    input_img = cv2.imread(str(sample["input_image"]))
    pred_img = cv2.imread(str(sample["pred_image"]))
    mask = combine_masks(sample["mask_images"])
    if input_img is None or pred_img is None or mask is None:
        return None

    aligned_img = align_images(input_img, pred_img)
    if aligned_img is None:
        return None

    mse, psnr = calculate_masked_metrics(input_img, aligned_img, mask)
    if mse is None or psnr is None:
        return None

    return {
        "task_id": sample["task_id"],
        "id_str": sample["id_str"],
        "lang": sample["lang"],
        "operation": sample["operation"],
        "mse": round(mse, 4),
        "psnr": round(psnr, 4),
        "input_image": sample["input_image"].name,
        "pred_image": sample["pred_image"].name,
        "mask_images": [path.name for path in sample["mask_images"]],
        "alignment_status": "success",
    }


def evaluate_samples(samples: List[Dict], workers: int) -> Dict:
    results: List[Dict] = []
    failed_count = 0

    if workers <= 1:
        for sample in tqdm(samples, desc="Processing"):
            try:
                result = process_sample(sample)
            except Exception as exc:  # pragma: no cover - defensive runtime path
                logger.error("Error processing %s: %s", sample["task_id"], exc)
                result = None

            if result is None:
                failed_count += 1
            else:
                results.append(result)
        return build_output_data(results, METRIC_NAMES, failed_count)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_sample, sample): sample
            for sample in samples
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive runtime path
                logger.error("Error processing %s: %s", sample["task_id"], exc)
                result = None

            if result is None:
                failed_count += 1
            else:
                results.append(result)

    return build_output_data(results, METRIC_NAMES, failed_count)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked MSE/PSNR evaluation: compares prediction vs source image on the non-edited background only."
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=sorted(CATEGORY_SIZES.keys()),
        help="Dataset category (Art/Event/Fashion/Food/Quotes)",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the dataset category directory (e.g. dataset/Quotes)",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Path to the model prediction category directory (e.g. predictions/<model>/Quotes)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument("--start_id", type=int, default=1, help="Starting sample ID")
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="Ending sample ID (default: full size of the chosen category)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes (e.g. en,zh,ja). Defaults to all 12 languages.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    category = args.category
    if args.end_id is None:
        args.end_id = CATEGORY_SIZES[category]

    languages = None
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    samples = scan_dataset(
        input_dir=args.input_dir,
        pred_dir=args.pred_dir,
        category=category,
        start_id=args.start_id,
        end_id=args.end_id,
        languages=languages,
    )
    logger.info("Found %s samples to process", len(samples))

    logger.info("Using workers: %s", args.workers)
    output_data = evaluate_samples(samples, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("MASKED MSE/PSNR EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Samples    : {output_data['summary']['total_samples']}")
    print(f"Failed Samples   : {output_data['summary']['failed_samples']}")
    print(f"Average MSE      : {output_data['summary']['avg_mse']:.4f}")
    print(f"Average PSNR     : {output_data['summary']['avg_psnr']:.4f} dB")
    print("=" * 50)
    logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
