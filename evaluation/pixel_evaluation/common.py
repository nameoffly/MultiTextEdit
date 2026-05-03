#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


CATEGORY_SIZES = {
    "Art": 70,
    "Event": 60,
    "Fashion": 110,
    "Food": 30,
    "Quotes": 30,
}

DEFAULT_LANGUAGES = [
    "en",
    "zh",
    "bn",
    "ru",
    "he",
    "yo",
    "nl",
    "vi",
    "ko",
    "ja",
    "es",
    "ar",
]


def metadata_filename(category: str, id_str: str) -> str:
    return f"{category}_{id_str}.json"


def derive_mask_name(image_name: str) -> str:
    image_path = Path(image_name)
    return f"{image_path.stem}_mask.jpg"


def locate_sample_paths(
    *,
    category: str,
    id_str: str,
    lang: str,
    input_dir: Path,
    pred_dir: Path,
) -> Optional[Dict]:
    lang_dir = input_dir / id_str / lang
    pred_lang_dir = pred_dir / id_str / lang
    metadata_path = lang_dir / metadata_filename(category, id_str)

    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_image_name = metadata.get("input_image", "1.jpg")
    output_image_name = metadata.get("output_image")
    if not output_image_name:
        return None

    input_image = lang_dir / input_image_name
    pred_image = pred_lang_dir / f"{id_str}_edited.png"
    input_mask = lang_dir / derive_mask_name(input_image_name)
    output_mask = lang_dir / derive_mask_name(output_image_name)

    required_paths = [input_image, pred_image, input_mask, output_mask]
    if any(not path.exists() for path in required_paths):
        return None

    return {
        "task_id": f"TextEditing_{category}_{id_str}_{lang}",
        "category": category,
        "id_str": id_str,
        "lang": lang,
        "operation": metadata.get("editing_method", "unknown"),
        "metadata_path": metadata_path,
        "input_image": input_image,
        "pred_image": pred_image,
        "mask_images": [input_mask, output_mask],
        "output_image": lang_dir / output_image_name,
        "metadata": metadata,
    }


def scan_dataset(
    *,
    input_dir: Path,
    pred_dir: Path,
    category: str,
    start_id: int = 1,
    end_id: Optional[int] = None,
    languages: Optional[List[str]] = None,
) -> List[Dict]:
    if end_id is None:
        end_id = CATEGORY_SIZES.get(category, 30)
    if languages is None:
        languages = DEFAULT_LANGUAGES

    samples: List[Dict] = []
    for id_num in range(start_id, end_id + 1):
        id_str = f"{id_num:03d}"
        for lang in languages:
            sample = locate_sample_paths(
                category=category,
                id_str=id_str,
                lang=lang,
                input_dir=input_dir,
                pred_dir=pred_dir,
            )
            if sample is not None:
                samples.append(sample)

    return samples


def _average(values: List[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def compute_statistics(results: List[Dict], metric_names: List[str]) -> Dict:
    if not results:
        summary = {f"avg_{metric}": 0.0 for metric in metric_names}
        summary.update({"total_samples": 0, "failed_samples": 0})
        return {
            "summary": summary,
            "by_operation": {},
            "by_language": {},
        }

    summary = {
        f"avg_{metric}": _average([result[metric] for result in results])
        for metric in metric_names
    }
    summary.update({"total_samples": len(results), "failed_samples": 0})

    def aggregate(group_key: str) -> Dict[str, Dict]:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for result in results:
            grouped[result[group_key]].append(result)

        aggregates: Dict[str, Dict] = {}
        for key, items in grouped.items():
            entry = {
                f"avg_{metric}": _average([item[metric] for item in items])
                for metric in metric_names
            }
            entry["count"] = len(items)
            aggregates[key] = entry
        return aggregates

    return {
        "summary": summary,
        "by_operation": aggregate("operation"),
        "by_language": aggregate("lang"),
    }


def build_output_data(results: List[Dict], metric_names: List[str], failed_count: int) -> Dict:
    output = compute_statistics(results, metric_names)
    output["summary"]["failed_samples"] = failed_count
    output["details"] = results
    return output
