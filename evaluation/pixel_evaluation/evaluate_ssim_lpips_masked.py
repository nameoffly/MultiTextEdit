#!/usr/bin/env python3

import argparse
import json
import logging
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


METRIC_NAMES = ["ssim", "lpips"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def require_image_metric_dependencies():
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "evaluate_ssim_lpips_masked.py requires opencv-python, numpy, pillow, torch, and torchmetrics to run."
        ) from exc

    return cv2, np, torch, Image, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity


class MetricsCalculator:
    def __init__(self, device: str) -> None:
        _, _, _, _, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity = (
            require_image_metric_dependencies()
        )
        self.device = device
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def _prepare_tensor(self, image, mask=None):
        _, np, torch, _, _, _ = require_image_metric_dependencies()
        array = np.array(image).astype(np.float32) / 255.0

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, None]
            if mask.max() > 1.0:
                mask = mask / 255.0
            array = array * mask

        return torch.tensor(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def calculate_metrics(self, pred_image, gt_image, mask) -> Tuple[float, float]:
        pred_tensor = self._prepare_tensor(pred_image, mask)
        gt_tensor = self._prepare_tensor(gt_image, mask)
        ssim = self.ssim_metric(pred_tensor, gt_tensor).cpu().item()
        lpips = self.lpips_metric(pred_tensor * 2 - 1, gt_tensor * 2 - 1).cpu().item()
        return ssim, lpips


def load_rgb_image(path: Path):
    _, _, _, Image, _, _ = require_image_metric_dependencies()
    return Image.open(path).convert("RGB")


def load_mask_images(mask_paths: List[Path]):
    cv2, _, _, _, _, _ = require_image_metric_dependencies()

    masks = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        masks.append(mask)
    return masks


def normalize_sample_dimensions(input_image, pred_image, masks):
    cv2, _, _, Image, _, _ = require_image_metric_dependencies()

    target_size = input_image.size
    if pred_image.size != target_size:
        pred_image = pred_image.resize(target_size, Image.Resampling.BILINEAR)

    target_width, target_height = target_size
    normalized_masks = []
    for mask in masks:
        if mask.shape[:2] != (target_height, target_width):
            mask = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        normalized_masks.append(mask)

    return input_image, pred_image, normalized_masks


def combine_background_mask(masks):
    cv2, np, _, _, _, _ = require_image_metric_dependencies()

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return (combined_mask < 128).astype(np.uint8) * 255


def process_sample(sample: Dict, device: str) -> Optional[Dict]:
    input_image = load_rgb_image(sample["input_image"])
    pred_image = load_rgb_image(sample["pred_image"])
    masks = load_mask_images(sample["mask_images"])
    if masks is None:
        return None

    input_image, pred_image, masks = normalize_sample_dimensions(
        input_image,
        pred_image,
        masks,
    )
    background_mask = combine_background_mask(masks)
    if background_mask is None:
        return None

    calculator = MetricsCalculator(device)
    ssim, lpips = calculator.calculate_metrics(pred_image, input_image, background_mask)

    return {
        "task_id": sample["task_id"],
        "id_str": sample["id_str"],
        "lang": sample["lang"],
        "operation": sample["operation"],
        "ssim": round(ssim, 4),
        "lpips": round(lpips, 4),
        "input_image": sample["input_image"].name,
        "pred_image": sample["pred_image"].name,
        "mask_images": [path.name for path in sample["mask_images"]],
    }


def evaluate_samples(samples: List[Dict], device: str, workers: int) -> Dict:
    results: List[Dict] = []
    failed_count = 0

    if workers <= 1:
        for sample in tqdm(samples, desc="Processing"):
            try:
                result = process_sample(sample, device)
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
            executor.submit(process_sample, sample, device): sample
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
        description="Masked SSIM/LPIPS evaluation: compares prediction vs source image on the non-edited background only."
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
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    category = args.category
    if args.end_id is None:
        args.end_id = CATEGORY_SIZES[category]

    try:
        _, _, torch, _, _, _ = require_image_metric_dependencies()
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"
    except RuntimeError:
        pass

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
    logger.info("Using device: %s", args.device)

    output_data = evaluate_samples(samples, args.device, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("MASKED SSIM/LPIPS EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Samples    : {output_data['summary']['total_samples']}")
    print(f"Failed Samples   : {output_data['summary']['failed_samples']}")
    print(f"Average SSIM     : {output_data['summary']['avg_ssim']:.4f}")
    print(f"Average LPIPS    : {output_data['summary']['avg_lpips']:.4f}")
    print("=" * 50)
    logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()
