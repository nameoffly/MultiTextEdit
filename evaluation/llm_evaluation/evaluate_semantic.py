"""
Semantic evaluation script for multilingual image text editing.

A multimodal LVM judge (default: gpt-5.4 via the OpenAI Responses API) scores
each prediction across six dimensions:
- IF (Instruction Following)
- TA (Text Accuracy)
- VC (Visual Consistency)
- LP (Layout Preservation)
- SE (Semantic Expectation, only when knowledge_prompt is provided)
- LSF (Language/Script Fidelity, two-stage trace + score)

The script is model-agnostic for the system under evaluation: any model whose
predictions follow the directory layout below can be evaluated.

    pred_dir/{ID}/{lang}/{ID}_edited.png
"""

import os
import sys
import logging
import json
import base64
import argparse
import time
import random
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional
from threading import Lock

from dotenv import load_dotenv
from tqdm import tqdm
import requests as http_requests

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GPT5.4Evaluator")

# --- Load Environment ---
load_dotenv()

# --- API Configuration ---
def setup_config(require_key: bool = True):
    """Load API configuration from environment.

    When require_key is True the script exits if OPENAI_API_KEY is missing.
    The actual exit only happens at run time (inside main), so --help works
    without any API key configured.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if require_key and not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    api_url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1/responses')
    model = os.getenv('OPENAI_MODEL', 'gpt-5.4')
    timeout = float(os.getenv('OPENAI_TIMEOUT', 900.0))

    if api_key:
        logger.info(f"API configured: model={model}, url={api_url}")
    return api_key, api_url, model, timeout

# Read configuration without enforcing the key, so --help works without .env.
# The key is enforced again inside main() before any actual API call.
API_KEY, API_URL, API_MODEL, API_TIMEOUT = setup_config(require_key=False)
VALID_RESPONSE_MODES = {"auto", "nonstream", "stream"}
RESPONSE_MODE = os.getenv("OPENAI_RESPONSE_MODE", "auto").strip().lower() or "auto"
if RESPONSE_MODE not in VALID_RESPONSE_MODES:
    logger.warning(f"Unknown OPENAI_RESPONSE_MODE={RESPONSE_MODE!r}, falling back to 'auto'")
    RESPONSE_MODE = "auto"

# --- Evaluation Prompts (from TextEditBench) ---
PROMPTS = {
    "IF": """You are a professional image editing evaluation specialist.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- editing_method: the specific operation type (color_change | exchange | insert | relocation | scaling | text_delete)
- output_image: the expected perfect output image (a human-crafted reference for comparison, not the model's actual result)

Your job:
Evaluate **instruction_following (IF)** on a 0–5 scale. Focus only on whether the pred_image performs exactly the requested operation type (editing_method) and nothing else.
Do **not** reward or penalize aspects unrelated to instruction compliance.

Penalty clause:
If pred_image performs wrong operation type, multiple operations when only one requested, or fails to execute the specified operation, subtract points accordingly.

Scoring Scale (0–5):
5 = Exactly the specified operation, no extra changes
4 = Correct operation with one tiny extra modification
3 = Generally correct but some unnecessary additions/omissions
2 = Operation attempted but with significant extra changes
1 = Wrong operation type or major execution errors
0 = No detectable change or completely wrong approach

Examples:
- Example 5: Instruction "Delete the word 'SALE'" with editing_method "text_delete" and pred_image shows only that word removed. Score 5.
- Example 3: Deleted "SALE" correctly but also changed font of other text when not requested. Score 3.
- Example 1: Used "exchange" (changed SALE to something else) instead of "text_delete", or deleted wrong text. Score 1.

Reasoning Steps:
1. Identify the requested operation from prompt and editing_method.
2. Compare pred_image with input_image and output_image.
3. Check if only the specified operation was performed.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"IF": X, "rationale": "short explanation of why this score"}""",

    "TA": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- output_image: the expected perfect output image (a human-crafted reference for comparison, not the model's actual result)

Your job:
Evaluate **text_accuracy (TA)** on a 0–5 scale. Focus only on whether the text content in pred_image matches the instruction and the expected output_image.
Do **not** reward or penalize aspects unrelated to text accuracy (e.g., font style, layout, background).

Penalty clause:
If the text content is wrong, misspelled, mistranslated, or unchanged when should change, subtract points accordingly.

Scoring Scale (0–5):
5 = Target text fully correct in content/spelling/case
4 = Correct but with one minor spelling/formatting inconsistency
3 = Core text mostly correct but noticeable error remains
2 = Several text elements wrong or inconsistent
1 = Text largely incorrect or unchanged when should change
0 = Completely wrong text or no change

Examples:
- Example 5: Instruction "Change 'Sale' to 'Sold Out'" and pred_image shows exactly "Sold Out". Score 5.
- Example 4: Shows "Sold out" with lowercase 'o'. Score 4.
- Example 3: Shows "Sold Oat" instead of "Sold Out". Score 3.
- Example 1: Still shows "Sale" unchanged. Score 1.

Reasoning Steps:
1. Identify the requested text change from prompt.
2. Extract/compare text from input_image (before) and pred_image (after).
3. Compare pred_image text with expected output_image text.
4. Check text content accuracy strictly against instruction.
5. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"TA": X, "rationale": "short explanation of why this score"}""",

    "VC": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- output_image: the expected perfect output image (a human-crafted reference for comparison, not the model's actual result)

Your job:
Evaluate **visual_consistency (VC)** on a 0–5 scale. Focus only on whether the new or edited elements visually integrate with the rest of the image in font/weight, alignment/perspective, edge anti-aliasing, color/lighting blending.
Do **not** reward or penalize unrelated content changes.

Penalty clause:
If new text or objects look pasted, misaligned, haloed, or style-mismatched, subtract points.

Scoring Scale (0–5):
5 = Perfect integration: matching font, size, color, alignment; no visible artifacts
4 = Very good integration: only tiny mismatch
3 = Moderate integration: visible mismatch (slight halo, mild misalignment)
2 = Poor integration: obvious halo or style mismatch
1 = Very poor integration: pasted look or misaligned
0 = Completely inconsistent/unreadable

Examples:
- Example 5: New text matches surrounding sign perfectly in font, color, and perspective. Score 5.
- Example 3: New text correct content but with mild white halo around edges. Score 3.
- Example 1: New text obviously pasted with wrong color and misaligned. Score 1.

Reasoning Steps:
1. Identify edited elements in pred_image.
2. Compare their visual integration quality to input_image and output_image.
3. Check for artifacts, misalignment, or style mismatches.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"VC": X, "rationale": "short explanation of why this score"}""",

    "LP": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- output_image: the expected perfect output image (a human-crafted reference for comparison, not the model's actual result)

Your job:
Evaluate **layout_preservation (LP)** on a 0–5 scale. Focus only on whether non-target areas remained unchanged compared to input_image.
Do **not** reward or penalize the quality of instructed changes themselves.

Penalty clause:
If pred_image alters background, other objects or layout unnecessarily, subtract points.

Scoring Scale (0–5):
5 = All non-target areas unchanged
4 = Almost unchanged, only tiny disturbance
3 = Minor unrelated disturbance
2 = Several unrelated changes
1 = Large portions altered unnecessarily
0 = Layout completely different

Examples:
- Example 5: Only target text changed from "Open" to "Closed", rest of sign and background untouched. Score 5.
- Example 3: Target text changed correctly but background slightly shifted or blurred. Score 3.
- Example 1: New objects added and overall layout composition different. Score 1.

Reasoning Steps:
1. Identify target and non-target areas from prompt.
2. Compare pred_image with input_image and output_image.
3. Detect any unrelated changes to non-target areas.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"LP": X, "rationale": "short explanation of why this score"}""",

    "SE": """You are an expert evaluator of text-in-image editing.

You will receive for each sample:
- input_image: the original unedited image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- knowledge_prompt: semantic expectation or logical consequence
- output_image: the expected perfect output image (a human-crafted reference for comparison, not the model's actual result)

Your job:
Evaluate **semantic_expectation (SE)** on a 0–5 scale.
Use the knowledge_prompt to understand linked expectations and check whether these are reflected in pred_image.
Focus only on whether these linked expectations are satisfied.
Do **not** reward or penalize unrelated features.

Penalty clause:
If pred_image fails to reflect expected linked changes or contradicts knowledge_prompt, subtract points.

Scoring Scale (0–5):
5 = Fully matches all linked/knowledge expectations
4 = Matches but with one small linked detail missed
3 = Partially satisfies: core effect present but significant linked effect missing
2 = Several linked effects missing or inconsistent
1 = Contradicts or ignores knowledge expectation
0 = No evidence of linked effect

Examples:
- Example 5: Instruction "Remove chili from dish" + knowledge_prompt "Removing 'chili' implies spicy icon should be absent". Pred_image shows dish without chili and spicy icon gone. Score 5.
- Example 3: Chili removed correctly but spicy icon still present. Score 3.
- Example 1: Chili still present and spicy icon still present, ignoring both instruction and semantic expectation. Score 1.

Reasoning Steps:
1. Identify instruction and linked expectations from prompt + knowledge_prompt.
2. Compare pred_image with input_image and output_image.
3. Check whether linked expectations are satisfied beyond direct instruction.
4. Assign score 0–5 using the scale and examples.

Output strictly in JSON:
{"SE": X, "rationale": "short explanation of why this score"}"""
}

# --- LSF Prompts ---
LSF_TRACE_PROMPT = """You are an expert evaluator of multilingual text-in-image editing.

Your job is to identify ONLY the edited target text for language/script fidelity evaluation.

You will receive:
- input_image: the original unedited image
- output_image: the expected perfect output image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- editing_method: the specific operation type
- language metadata: expected script, writing direction, and language-sensitive features

Rules:
1. Compare input_image and output_image FIRST to determine which text was edited.
2. Ignore all untouched text, even if it is larger or easier to read.
3. For relocation or scaling, match target text by text identity and edit role, not by absolute position.
4. For exchange, capture both the original text and expected replacement text.
5. For insert, input_text_before may be null.
6. For text_delete, return trace_status = "not_applicable".
7. Do NOT score quality in this stage. Only isolate and transcribe the target text.

Output strictly in JSON with this schema:
{
  "trace_status": "success|ambiguous|not_applicable|error",
  "not_applicable_reason": "text_delete|null",
  "overall_trace_confidence": 0.0,
  "target_segments": [
    {
      "segment_id": "seg_1",
      "edit_role": "style_changed|replaced|inserted|relocated|scaled|unknown",
      "reference_locator": "brief location description",
      "input_text_before": "text or null",
      "output_text_expected": "target text in output image",
      "pred_text_observed": "corresponding text in pred image",
      "pred_found": true,
      "trace_confidence": 0.0,
      "notes": "brief note"
    }
  ],
  "ignored_non_target_text": true,
  "trace_rationale": "brief explanation"
}"""

LSF_SCORE_PROMPT = """You are an expert evaluator of language/script fidelity in edited text images.

You will receive:
- input_image: the original unedited image
- output_image: the expected perfect output image
- pred_image: the candidate image produced by the model under test
- prompt: the user editing instruction
- editing_method: the specific operation type
- language metadata: expected script, writing direction, and language-sensitive features
- trace_json: the target text trace from stage 1

Your job:
Score ONLY the target_segments listed in trace_json for language/script fidelity (LSF).

Focus on:
- character substitutions, omissions, or insertions
- missing or incorrect diacritics, tone marks, accents, or script-specific marks
- RTL/LTR ordering problems
- punctuation or bracket direction problems
- script mixing
- language-specific typography or shaping problems

Rules:
1. Re-check the images, but do NOT expand beyond the traced target text.
2. Ignore untouched text entirely.
3. If trace_status is not "success", or trace confidence is below 0.60, return LSF_status = "unscorable".
4. If the target text should exist but is missing in pred_image, treat that as a severe failure.
5. For text_delete, return LSF_status = "not_applicable".

Scoring scale:
5 = Fully correct script/language rendering
4 = One minor script-level issue
3 = Noticeable character or mark-level issue
2 = Multiple clear issues or one major direction/script problem
1 = Most of the target text is wrong
0 = Missing, unreadable, or clearly wrong script/direction

Output strictly in JSON with this schema:
{
  "LSF_status": "scored|not_applicable|unscorable|error",
  "LSF": 0,
  "error_tags": ["tone_mark_error"],
  "judge_confidence": 0.0,
  "per_segment": [
    {
      "segment_id": "seg_1",
      "segment_score": 0,
      "hard_fail": false,
      "error_tags": ["tone_mark_error"],
      "reason": "brief explanation"
    }
  ],
  "rationale": "brief explanation"
}"""

# --- Language Codes ---
LANGUAGES = ['en', 'bn', 'ru', 'he', 'yo', 'nl', 'vi', 'ko', 'ja', 'es', 'ar', 'zh']
BASE_DIMENSIONS = ["IF", "TA", "VC", "LP"]
REPORT_DIMENSIONS = ["IF", "TA", "VC", "LP", "SE", "LSF"]
LSF_TRACE_STATUSES = {"success", "ambiguous", "not_applicable", "error"}
LSF_COMPLETE_STATUSES = {"scored", "not_applicable", "unscorable"}
LSF_SCORE_STATUSES = LSF_COMPLETE_STATUSES | {"error"}
LSF_ERROR_TAGS = {
    "target_text_missing",
    "character_substitution",
    "character_missing",
    "character_extra",
    "diacritic_error",
    "tone_mark_error",
    "accent_error",
    "rtl_ltr_order_error",
    "punctuation_direction_error",
    "script_mixing",
    "script_typography_error",
}
LANGUAGE_METADATA = {
    "en": {"expected_script": "Latin", "expected_direction": "LTR", "script_sensitive_features": ["basic Latin letter fidelity"]},
    "es": {"expected_script": "Latin", "expected_direction": "LTR", "script_sensitive_features": ["accent marks and punctuation"]},
    "nl": {"expected_script": "Latin", "expected_direction": "LTR", "script_sensitive_features": ["Latin spelling fidelity"]},
    "yo": {"expected_script": "Latin", "expected_direction": "LTR", "script_sensitive_features": ["tone marks", "underdot characters"]},
    "vi": {"expected_script": "Latin", "expected_direction": "LTR", "script_sensitive_features": ["tone marks", "diacritics"]},
    "ru": {"expected_script": "Cyrillic", "expected_direction": "LTR", "script_sensitive_features": ["Cyrillic letter fidelity"]},
    "bn": {"expected_script": "Bengali", "expected_direction": "LTR", "script_sensitive_features": ["Bengali marks", "conjuncts", "matras"]},
    "zh": {"expected_script": "Han", "expected_direction": "LTR", "script_sensitive_features": ["Chinese character fidelity"]},
    "ja": {"expected_script": "Japanese mixed script", "expected_direction": "LTR", "script_sensitive_features": ["kanji", "hiragana", "katakana"]},
    "ko": {"expected_script": "Hangul", "expected_direction": "LTR", "script_sensitive_features": ["Hangul syllable fidelity"]},
    "ar": {"expected_script": "Arabic", "expected_direction": "RTL", "script_sensitive_features": ["RTL order", "Arabic letter shaping", "punctuation direction"]},
    "he": {"expected_script": "Hebrew", "expected_direction": "RTL", "script_sensitive_features": ["RTL order", "Hebrew punctuation direction"]},
}

# --- Helper Functions ---

def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None

def get_language_metadata(lang: str) -> Dict[str, Any]:
    """Return the expected script metadata for a language code."""
    default = {
        "expected_script": "Unknown",
        "expected_direction": "LTR",
        "script_sensitive_features": ["character fidelity"],
    }
    metadata = LANGUAGE_METADATA.get(lang, default)
    return {
        "expected_script": metadata["expected_script"],
        "expected_direction": metadata["expected_direction"],
        "script_sensitive_features": list(metadata["script_sensitive_features"]),
    }

def clamp_probability(value: Any, default: float = 0.0) -> float:
    """Clamp a confidence-like value into the 0..1 range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))

def coerce_score(value: Any) -> Optional[float]:
    """Convert a score-like value into a numeric 0..5 score."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= numeric <= 5.0:
        return numeric
    return None

def normalize_error_tags(value: Any) -> List[str]:
    """Normalize LSF error tags into the supported vocabulary."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [tag for tag in value if isinstance(tag, str) and tag in LSF_ERROR_TAGS]

def normalize_optional_text(value: Any) -> Optional[str]:
    """Normalize optional textual values."""
    if value is None:
        return None
    return str(value)

def build_common_user_content(input_b64: str, pred_b64: str, output_b64: str,
                              prompt: str, editing_method: str = None,
                              knowledge_prompt: str = None,
                              extra_text_items: Optional[List[Tuple[str, Any]]] = None,
                              schema_hint: str = None) -> List[Dict[str, str]]:
    """Build the common user payload for Responses API image-eval requests."""
    user_content = [
        {"type": "input_text", "text": f"User editing instruction (prompt): {prompt}"}
    ]
    if editing_method:
        user_content.append({"type": "input_text", "text": f"Editing method: {editing_method}"})
    if knowledge_prompt:
        user_content.append({"type": "input_text", "text": f"Knowledge prompt: {knowledge_prompt}"})
    if extra_text_items:
        for label, value in extra_text_items:
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, tuple):
                value = ", ".join(str(item) for item in value)
            user_content.append({"type": "input_text", "text": f"{label}: {value}"})

    user_content.extend([
        {"type": "input_text", "text": "Input image (original):"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{input_b64}"},
        {"type": "input_text", "text": "Pred image (model output):"},
        {"type": "input_image", "image_url": f"data:image/png;base64,{pred_b64}"},
        {"type": "input_text", "text": "Output image (expected reference):"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{output_b64}"},
    ])
    if schema_hint:
        user_content.append({"type": "input_text", "text": schema_hint})
    return user_content

def extract_output_text_from_response_json(resp_json: Dict[str, Any]) -> str:
    """Extract the first output_text string from a standard Responses payload."""
    for output_item in resp_json.get("output", []):
        if output_item.get("type") == "message":
            for content_item in output_item.get("content", []):
                if content_item.get("type") == "output_text":
                    return content_item.get("text", "")
    return ""

def extract_output_text_from_stream_lines(lines) -> str:
    """Reconstruct output_text from a streamed Responses SSE payload."""
    delta_parts: List[str] = []
    final_text = ""

    for raw_line in lines:
        if raw_line is None:
            continue

        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue

        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                delta_parts.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str):
                final_text = text
        elif event_type == "response.content_part.done":
            part = event.get("part", {})
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                final_text = part["text"]
        elif event_type == "response.output_item.done":
            item = event.get("item", {})
            item_text = extract_output_text_from_response_json({"output": [item]})
            if item_text:
                final_text = item_text

    return final_text or "".join(delta_parts)

def request_response_text(payload: Dict[str, Any], use_stream: bool = False) -> str:
    """Submit a Responses API request and extract text from JSON or SSE."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    request_payload = copy.deepcopy(payload)
    request_kwargs = {
        "headers": headers,
        "json": request_payload,
        "timeout": API_TIMEOUT,
    }

    if use_stream:
        request_payload["stream"] = True
        request_kwargs["stream"] = True

    resp = http_requests.post(API_URL, **request_kwargs)
    resp.raise_for_status()

    if use_stream:
        return extract_output_text_from_stream_lines(resp.iter_lines(decode_unicode=True))

    return extract_output_text_from_response_json(resp.json())

def call_llm_json(request_name: str, instructions: str, user_content: List[Dict[str, str]],
                  validator, error_result: Dict[str, Any], retry_count: int = 3,
                  max_output_tokens: int = 500) -> Dict[str, Any]:
    """Call the Responses API and return a parsed JSON object."""
    for attempt in range(retry_count):
        try:
            payload = {
                "model": API_MODEL,
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "temperature": 0.0,
                "max_output_tokens": max_output_tokens,
            }
            result_text = ""

            if RESPONSE_MODE in {"nonstream", "auto"}:
                result_text = request_response_text(payload, use_stream=False)
                if not result_text:
                    if RESPONSE_MODE == "auto":
                        logger.warning(
                            f"Empty non-stream response for {request_name}, trying stream fallback ({attempt + 1}/{retry_count})"
                        )
                    else:
                        logger.warning(f"Empty response for {request_name}, retrying ({attempt + 1}/{retry_count})")
                        continue

            if not result_text and RESPONSE_MODE in {"stream", "auto"}:
                result_text = request_response_text(payload, use_stream=True)
                if not result_text:
                    logger.warning(f"Empty stream response for {request_name}, retrying ({attempt + 1}/{retry_count})")
                    continue

            result_text = result_text.strip().replace("```json", "").replace("```", "").strip()
            start = result_text.find("{")
            end = result_text.rfind("}")
            if start != -1 and end != -1:
                result_text = result_text[start:end + 1]

            result_json = json.loads(result_text)
            if validator(result_json):
                return result_json

            logger.warning(f"Invalid JSON shape for {request_name}, retrying ({attempt + 1}/{retry_count})")

        except http_requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                if attempt == retry_count - 1:
                    break
                delay = 2.0 * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"{request_name} rate limited. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            elif attempt == retry_count - 1:
                break
            else:
                logger.warning(f"{request_name} HTTP error {status}, retrying ({attempt + 1}/{retry_count})")
                time.sleep(2.0)

        except http_requests.exceptions.Timeout:
            if attempt == retry_count - 1:
                break
            delay = 2.0 * (attempt + 1)
            logger.warning(f"{request_name} timeout. Retrying in {delay:.1f}s...")
            time.sleep(delay)

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {request_name}: {e}")
            if attempt == retry_count - 1:
                break
            time.sleep(2.0)

    fallback = copy.deepcopy(error_result)
    if request_name not in {"LSF_TRACE", "LSF_SCORE"}:
        fallback.setdefault("rationale", f"{request_name} request failed")
    return fallback

def is_valid_dimension_result(result: Any, dimension: str) -> bool:
    """Check whether a classic IF/TA/VC/LP/SE result is structurally valid."""
    return (
        isinstance(result, dict)
        and dimension in result
        and coerce_score(result.get(dimension)) is not None
        and isinstance(result.get("rationale"), str)
    )

def validate_lsf_trace_response(result: Any) -> bool:
    """Validate the raw JSON schema of an LSF trace response."""
    return isinstance(result, dict) and result.get("trace_status") in LSF_TRACE_STATUSES

def validate_lsf_score_response(result: Any) -> bool:
    """Validate the raw JSON schema of an LSF score response."""
    return isinstance(result, dict) and (
        result.get("LSF_status") in LSF_SCORE_STATUSES or coerce_score(result.get("LSF")) is not None
    )

def normalize_lsf_trace_segment(segment: Any, index: int) -> Dict[str, Any]:
    """Normalize a single stage-1 target segment."""
    segment = segment if isinstance(segment, dict) else {}
    return {
        "segment_id": str(segment.get("segment_id") or f"seg_{index}"),
        "edit_role": str(segment.get("edit_role") or "unknown"),
        "reference_locator": str(segment.get("reference_locator") or ""),
        "input_text_before": normalize_optional_text(segment.get("input_text_before")),
        "output_text_expected": normalize_optional_text(segment.get("output_text_expected")) or "",
        "pred_text_observed": normalize_optional_text(segment.get("pred_text_observed")),
        "pred_found": bool(segment.get("pred_found", False)),
        "trace_confidence": clamp_probability(segment.get("trace_confidence"), default=0.0),
        "notes": str(segment.get("notes") or ""),
    }

def normalize_lsf_trace(raw_trace: Any, editing_method: str) -> Dict[str, Any]:
    """Normalize a raw stage-1 LSF trace response."""
    raw_trace = raw_trace if isinstance(raw_trace, dict) else {}
    rationale = str(raw_trace.get("trace_rationale") or raw_trace.get("rationale") or "")

    if editing_method == "text_delete":
        return {
            "trace_version": "lsf-trace-v1",
            "trace_status": "not_applicable",
            "not_applicable_reason": "text_delete",
            "overall_trace_confidence": 1.0,
            "target_segments": [],
            "ignored_non_target_text": True,
            "trace_rationale": rationale or "text_delete samples do not have surviving target text for LSF.",
        }

    trace_status = raw_trace.get("trace_status")
    if trace_status not in LSF_TRACE_STATUSES:
        trace_status = "ambiguous"

    target_segments = raw_trace.get("target_segments", [])
    if not isinstance(target_segments, list):
        target_segments = []
    normalized_segments = [
        normalize_lsf_trace_segment(segment, index + 1)
        for index, segment in enumerate(target_segments)
    ]

    if trace_status == "success" and not normalized_segments:
        trace_status = "ambiguous"

    return {
        "trace_version": "lsf-trace-v1",
        "trace_status": trace_status,
        "not_applicable_reason": raw_trace.get("not_applicable_reason") if trace_status == "not_applicable" else None,
        "overall_trace_confidence": clamp_probability(raw_trace.get("overall_trace_confidence"), default=0.0),
        "target_segments": normalized_segments,
        "ignored_non_target_text": bool(raw_trace.get("ignored_non_target_text", True)),
        "trace_rationale": rationale or "Trace response did not include a rationale.",
    }

def normalize_lsf_score_segment(segment: Any, index: int) -> Dict[str, Any]:
    """Normalize a single stage-2 per-segment score."""
    segment = segment if isinstance(segment, dict) else {}
    return {
        "segment_id": str(segment.get("segment_id") or f"seg_{index}"),
        "segment_score": coerce_score(segment.get("segment_score")),
        "hard_fail": bool(segment.get("hard_fail", False)),
        "error_tags": normalize_error_tags(segment.get("error_tags")),
        "reason": str(segment.get("reason") or ""),
    }

def normalize_lsf_score(raw_score: Any, trace: Dict[str, Any], editing_method: str) -> Dict[str, Any]:
    """Normalize a raw stage-2 LSF score response."""
    raw_score = raw_score if isinstance(raw_score, dict) else {}
    rationale = str(raw_score.get("rationale") or "")

    if editing_method == "text_delete" or trace.get("trace_status") == "not_applicable":
        return {
            "lsf_version": "lsf-v1",
            "LSF_status": "not_applicable",
            "LSF": None,
            "error_tags": [],
            "judge_confidence": 1.0,
            "per_segment": [],
            "rationale": rationale or "text_delete samples are not scored for LSF.",
        }

    if trace.get("trace_status") == "error":
        return {
            "lsf_version": "lsf-v1",
            "LSF_status": "error",
            "LSF": None,
            "error_tags": [],
            "judge_confidence": 0.0,
            "per_segment": [],
            "rationale": rationale or "LSF trace generation failed.",
        }

    if trace.get("trace_status") != "success" or trace.get("overall_trace_confidence", 0.0) < 0.60:
        return {
            "lsf_version": "lsf-v1",
            "LSF_status": "unscorable",
            "LSF": None,
            "error_tags": normalize_error_tags(raw_score.get("error_tags")),
            "judge_confidence": clamp_probability(raw_score.get("judge_confidence"), default=0.0),
            "per_segment": [],
            "rationale": rationale or "Trace was ambiguous or low-confidence, so LSF was not scored.",
        }

    lsf_status = raw_score.get("LSF_status")
    if lsf_status not in LSF_SCORE_STATUSES:
        lsf_status = "scored" if coerce_score(raw_score.get("LSF")) is not None else "error"

    normalized_per_segment = raw_score.get("per_segment", [])
    if not isinstance(normalized_per_segment, list):
        normalized_per_segment = []
    normalized_per_segment = [
        normalize_lsf_score_segment(segment, index + 1)
        for index, segment in enumerate(normalized_per_segment)
    ]

    score = coerce_score(raw_score.get("LSF")) if lsf_status == "scored" else None
    if lsf_status == "scored" and score is None:
        lsf_status = "error"

    return {
        "lsf_version": "lsf-v1",
        "LSF_status": lsf_status,
        "LSF": score if lsf_status == "scored" else None,
        "error_tags": normalize_error_tags(raw_score.get("error_tags")),
        "judge_confidence": clamp_probability(raw_score.get("judge_confidence"), default=0.0),
        "per_segment": normalized_per_segment,
        "rationale": rationale or "LSF score response did not include a rationale.",
    }

def is_valid_lsf_trace(trace: Any) -> bool:
    """Check whether a stored LSF trace is structurally valid."""
    return (
        isinstance(trace, dict)
        and trace.get("trace_status") in LSF_TRACE_STATUSES
        and isinstance(trace.get("target_segments", []), list)
        and isinstance(trace.get("trace_rationale"), str)
    )

def is_valid_lsf_result(result: Any, editing_method: str = None) -> bool:
    """Check whether a stored LSF score is structurally valid."""
    if not isinstance(result, dict):
        return False
    status = result.get("LSF_status")
    if status not in LSF_SCORE_STATUSES or not isinstance(result.get("rationale"), str):
        return False
    if status == "scored":
        return coerce_score(result.get("LSF")) is not None
    if status == "not_applicable":
        return editing_method == "text_delete"
    return True

def is_task_complete(result: Dict[str, Any], task_info: Dict[str, Any]) -> bool:
    """Return True if a task already has every required dimension, including LSF."""
    if result.get("status") != "success":
        return False

    evaluation_results = result.get("evaluation_results", {})
    required_dims = list(BASE_DIMENSIONS)
    if task_info.get("has_knowledge_prompt"):
        required_dims.append("SE")

    for dim in required_dims:
        if not is_valid_dimension_result(evaluation_results.get(dim), dim):
            return False

    trace = result.get("evaluation_traces", {}).get("LSF_TRACE")
    if not is_valid_lsf_trace(trace):
        return False

    lsf_result = evaluation_results.get("LSF")
    editing_method = task_info.get("editing_method")
    if not is_valid_lsf_result(lsf_result, editing_method):
        return False

    if editing_method == "text_delete":
        return (
            trace.get("trace_status") == "not_applicable"
            and lsf_result.get("LSF_status") == "not_applicable"
        )

    return lsf_result.get("LSF_status") in {"scored", "unscorable"}

def locate_images(id_str: str, lang: str, input_dir: Path, pred_dir: Path, category: str = "Quotes") -> Tuple[Path, Path, Path, Dict]:
    """
    Locate the three images and metadata for a task.
    Returns: (input_image, output_image, pred_image, metadata)
    """
    # Ground truth directory
    lang_dir = input_dir / id_str / lang

    # Load metadata
    metadata_file = lang_dir / f"{category}_{id_str}.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Input image: fixed as 1.jpg
    input_image = lang_dir / "1.jpg"

    # Output image: from metadata
    output_image_name = metadata.get("output_image", "")
    if not output_image_name:
        raise ValueError(f"output_image not specified in metadata: {metadata_file}")
    output_image = lang_dir / output_image_name

    # Prediction image: {ID}_edited.png
    pred_lang_dir = pred_dir / id_str / lang
    pred_image = pred_lang_dir / f"{id_str}_edited.png"

    # Validate all files exist
    for path, label in [(input_image, "Input"), (output_image, "Output"), (pred_image, "Prediction")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} image not found: {path}")

    return input_image, output_image, pred_image, metadata

def call_llm_evaluation(dimension: str, input_b64: str, pred_b64: str,
                        output_b64: str, prompt: str, editing_method: str = None,
                        knowledge_prompt: str = None, retry_count: int = 3) -> Dict:
    """Call gpt-5.4 via Responses API for a specific evaluation dimension."""

    if dimension not in PROMPTS:
        logger.error(f"Prompt for dimension {dimension} not defined.")
        return {dimension: 0, "rationale": "Prompt missing"}
    user_content = build_common_user_content(
        input_b64, pred_b64, output_b64, prompt,
        editing_method=editing_method,
        knowledge_prompt=knowledge_prompt if dimension == "SE" else None,
        schema_hint=(
            "\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object.**\n"
            + f'{{"{dimension}": <score 0-5>, "rationale": "<explanation>"}}'
        )
    )
    return call_llm_json(
        request_name=dimension,
        instructions=PROMPTS[dimension],
        user_content=user_content,
        validator=lambda result_json: is_valid_dimension_result(result_json, dimension),
        error_result={dimension: 0, "rationale": f"{dimension} request failed"},
        retry_count=retry_count,
        max_output_tokens=500,
    )

def call_lsf_trace(input_b64: str, pred_b64: str, output_b64: str, prompt: str,
                   editing_method: str, lang: str, retry_count: int = 3) -> Dict[str, Any]:
    """Run stage-1 LSF trace generation."""
    language_metadata = get_language_metadata(lang)
    user_content = build_common_user_content(
        input_b64, pred_b64, output_b64, prompt,
        editing_method=editing_method,
        extra_text_items=[
            ("Language code", lang),
            ("Expected script", language_metadata["expected_script"]),
            ("Expected direction", language_metadata["expected_direction"]),
            ("Script-sensitive features", ", ".join(language_metadata["script_sensitive_features"])),
            ("Stage objective", "Compare input_image and output_image first, isolate only the edited target text, and transcribe it without scoring."),
        ],
        schema_hint="\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object matching the LSF_TRACE schema.**",
    )
    return call_llm_json(
        request_name="LSF_TRACE",
        instructions=LSF_TRACE_PROMPT,
        user_content=user_content,
        validator=validate_lsf_trace_response,
        error_result={
            "trace_status": "error",
            "overall_trace_confidence": 0.0,
            "target_segments": [],
            "ignored_non_target_text": True,
            "trace_rationale": "LSF_TRACE request failed",
        },
        retry_count=retry_count,
        max_output_tokens=1200,
    )

def call_lsf_score(input_b64: str, pred_b64: str, output_b64: str, prompt: str,
                   editing_method: str, lang: str, trace: Dict[str, Any],
                   retry_count: int = 3) -> Dict[str, Any]:
    """Run stage-2 LSF scoring."""
    language_metadata = get_language_metadata(lang)
    user_content = build_common_user_content(
        input_b64, pred_b64, output_b64, prompt,
        editing_method=editing_method,
        extra_text_items=[
            ("Language code", lang),
            ("Expected script", language_metadata["expected_script"]),
            ("Expected direction", language_metadata["expected_direction"]),
            ("Script-sensitive features", ", ".join(language_metadata["script_sensitive_features"])),
            ("trace_json", trace),
        ],
        schema_hint="\n**CRITICAL REQUIREMENT: Your entire response must be ONLY a valid JSON object matching the LSF_SCORE schema.**",
    )
    return call_llm_json(
        request_name="LSF_SCORE",
        instructions=LSF_SCORE_PROMPT,
        user_content=user_content,
        validator=validate_lsf_score_response,
        error_result={
            "LSF_status": "error",
            "LSF": None,
            "error_tags": [],
            "judge_confidence": 0.0,
            "per_segment": [],
            "rationale": "LSF_SCORE request failed",
        },
        retry_count=retry_count,
        max_output_tokens=1000,
    )

def evaluate_lsf(input_b64: str, pred_b64: str, output_b64: str, prompt: str,
                 editing_method: str, lang: str, existing_trace: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate the two-stage LSF pipeline for a single task."""
    if editing_method == "text_delete":
        trace = normalize_lsf_trace(existing_trace, editing_method)
        return trace, normalize_lsf_score({}, trace, editing_method)

    trace = None
    if is_valid_lsf_trace(existing_trace) and existing_trace.get("trace_status") != "error":
        trace = normalize_lsf_trace(existing_trace, editing_method)
    else:
        trace = normalize_lsf_trace(
            call_lsf_trace(input_b64, pred_b64, output_b64, prompt, editing_method, lang),
            editing_method,
        )

    if trace.get("trace_status") == "error":
        return trace, normalize_lsf_score({"LSF_status": "error", "rationale": trace.get("trace_rationale")}, trace, editing_method)

    if trace.get("trace_status") != "success" or trace.get("overall_trace_confidence", 0.0) < 0.60:
        return trace, normalize_lsf_score(
            {"LSF_status": "unscorable", "rationale": "Trace was ambiguous or low-confidence, so LSF was not scored."},
            trace,
            editing_method,
        )

    score = normalize_lsf_score(
        call_lsf_score(input_b64, pred_b64, output_b64, prompt, editing_method, lang, trace),
        trace,
        editing_method,
    )
    return trace, score

def evaluate_task(task_info: Dict[str, Any], input_dir: Path, pred_dir: Path, category: str = "Quotes") -> Dict:
    """Evaluate a single task across all dimensions."""
    try:
        task_id = task_info['task_id']
        id_str = task_info['id_str']
        lang = task_info['lang']
        existing_result = task_info.get("existing_result", {})

        # Locate images and metadata
        input_image, output_image, pred_image, metadata = locate_images(
            id_str, lang, input_dir, pred_dir, category
        )

        # Extract task information
        prompt = metadata.get("prompt", "")
        editing_method = metadata.get("editing_method", "")
        knowledge_prompt = metadata.get("knowledge_prompt", None)

        # Encode images
        input_b64 = encode_image_to_base64(input_image)
        output_b64 = encode_image_to_base64(output_image)
        pred_b64 = encode_image_to_base64(pred_image)

        if not all([input_b64, output_b64, pred_b64]):
            raise ValueError("Failed to encode one or more images")

        # Evaluate all dimensions
        results = copy.deepcopy(existing_result.get("evaluation_results", {}))
        traces = copy.deepcopy(existing_result.get("evaluation_traces", {}))
        dims = list(BASE_DIMENSIONS)
        if knowledge_prompt:
            dims.append("SE")

        for dim in dims:
            if not is_valid_dimension_result(results.get(dim), dim):
                results[dim] = call_llm_evaluation(
                    dim, input_b64, pred_b64, output_b64, prompt,
                    editing_method=editing_method, knowledge_prompt=knowledge_prompt
                )

        existing_trace = traces.get("LSF_TRACE")
        existing_lsf = results.get("LSF")
        if editing_method == "text_delete":
            traces["LSF_TRACE"] = normalize_lsf_trace(existing_trace, editing_method)
            results["LSF"] = normalize_lsf_score(existing_lsf, traces["LSF_TRACE"], editing_method)
        else:
            trace_ready = is_valid_lsf_trace(existing_trace)
            score_ready = (
                is_valid_lsf_result(existing_lsf, editing_method)
                and existing_lsf.get("LSF_status") in {"scored", "unscorable"}
            )

            if trace_ready and score_ready:
                traces["LSF_TRACE"] = normalize_lsf_trace(existing_trace, editing_method)
                results["LSF"] = normalize_lsf_score(existing_lsf, traces["LSF_TRACE"], editing_method)
            else:
                traces["LSF_TRACE"], results["LSF"] = evaluate_lsf(
                    input_b64, pred_b64, output_b64, prompt, editing_method, lang,
                    existing_trace=existing_trace,
                )

        return {
            "status": "success",
            "task_id": task_id,
            "category": category,
            "model": existing_result.get("model", "gpt-image-1.5"),
            "id_str": id_str,
            "lang": lang,
            "operation": editing_method,
            "evaluation_results": results,
            "evaluation_traces": traces,
            "schema_version": "2.0",
        }

    except Exception as e:
        logger.error(f"Error evaluating task {task_id}: {e}")
        return {
            "status": "error",
            "task_id": task_id,
            "error": str(e)
        }

def process_task_wrapper(task_info: Dict, input_dir: Path, pred_dir: Path, category: str = "Quotes") -> Dict:
    """Wrapper for thread pool execution."""
    return evaluate_task(task_info, input_dir, pred_dir, category)

def scan_dataset(input_dir: Path, pred_dir: Path,
                        start_id: int = 1, end_id: int = 30,
                        languages: List[str] = None, category: str = "Quotes") -> List[Dict]:
    """Scan the dataset and collect all tasks."""
    if languages is None:
        languages = LANGUAGES

    tasks = []

    for id_num in range(start_id, end_id + 1):
        id_str = f"{id_num:03d}"

        for lang in languages:
            task_id = f"TextEditing_{category}_{id_str}_{lang}"

            # Check if prediction exists
            pred_file = pred_dir / id_str / lang / f"{id_str}_edited.png"
            if not pred_file.exists():
                logger.warning(f"Prediction not found for {task_id}, skipping")
                continue

            metadata_file = input_dir / id_str / lang / f"{category}_{id_str}.json"
            if not metadata_file.exists():
                logger.warning(f"Metadata not found for {task_id}, skipping")
                continue
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            tasks.append({
                'task_id': task_id,
                'id_str': id_str,
                'lang': lang,
                'editing_method': metadata.get("editing_method", ""),
                'has_knowledge_prompt': bool(metadata.get("knowledge_prompt")),
            })

    return tasks

def check_resume_state(tasks: List[Dict], output_file: Path) -> List[Dict]:
    """Check already evaluated tasks and return remaining tasks."""
    if not output_file.exists():
        return tasks

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)

        existing_by_task = {}
        for result in existing_results:
            if isinstance(result, dict) and result.get('task_id'):
                existing_by_task[result['task_id']] = result

        remaining = []
        skipped = 0
        for task in tasks:
            existing = existing_by_task.get(task['task_id'])
            if existing and is_task_complete(existing, task):
                skipped += 1
                continue

            task_copy = dict(task)
            if existing:
                task_copy['existing_result'] = existing
            remaining.append(task_copy)

        skipped = len(tasks) - len(remaining)

        if skipped > 0:
            logger.info(f"Resume mode: skipping {skipped} already evaluated tasks")

        return remaining

    except Exception as e:
        logger.warning(f"Could not read existing results: {e}")
        return tasks

def append_result_to_file(result: Dict, output_file: Path, lock: Lock):
    """Thread-safe upsert of a single result into the output file."""
    with lock:
        try:
            # Read existing results
            existing = []
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)

            updated = False
            for index, current in enumerate(existing):
                if current.get('task_id') == result.get('task_id'):
                    existing[index] = result
                    updated = True
                    break

            if not updated:
                existing.append(result)

            # Write back
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to append result to {output_file}: {e}")

def extract_dimension_score(evaluation_results: Dict[str, Any], dimension: str) -> Optional[float]:
    """Extract a numeric score for a reportable dimension."""
    if dimension == "LSF":
        lsf_result = evaluation_results.get("LSF", {})
        if isinstance(lsf_result, dict) and lsf_result.get("LSF_status") == "scored":
            return coerce_score(lsf_result.get("LSF"))
        return None

    dimension_result = evaluation_results.get(dimension, {})
    if isinstance(dimension_result, dict):
        return coerce_score(dimension_result.get(dimension))
    return None

def extract_lsf_status(evaluation_results: Dict[str, Any]) -> Optional[str]:
    """Extract the current LSF status from an evaluation result."""
    lsf_result = evaluation_results.get("LSF", {})
    if isinstance(lsf_result, dict):
        status = lsf_result.get("LSF_status")
        if status in LSF_SCORE_STATUSES:
            return status
    return None

def init_stats_bucket() -> Dict[str, Any]:
    """Create an aggregation bucket for operation/language stats."""
    return {
        "count": 0,
        "scores": {dim: [] for dim in REPORT_DIMENSIONS},
        "LSF_valid_count": 0,
        "LSF_not_applicable_count": 0,
        "LSF_unscorable_count": 0,
    }

def generate_statistics(results: List[Dict], output_dir: Path):
    """Generate statistics and reports from evaluation results."""
    if not results:
        logger.warning("No results to generate statistics")
        return

    # Filter successful results
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']

    total = len(results)
    success_count = len(successful)
    failed_count = len(failed)

    logger.info(f"\nEvaluation Summary:")
    logger.info(f"Total tasks: {total}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {success_count/total*100:.2f}%")

    if not successful:
        return

    overall_bucket = init_stats_bucket()
    by_operation = {}
    by_language = {}

    for r in successful:
        eval_res = r.get('evaluation_results', {})
        operation = r.get('operation', 'unknown')
        language = r.get('lang', 'unknown')

        overall_bucket["count"] += 1
        by_operation.setdefault(operation, init_stats_bucket())
        by_language.setdefault(language, init_stats_bucket())
        by_operation[operation]["count"] += 1
        by_language[language]["count"] += 1

        lsf_status = extract_lsf_status(eval_res)
        for bucket in (overall_bucket, by_operation[operation], by_language[language]):
            if lsf_status == "scored":
                bucket["LSF_valid_count"] += 1
            elif lsf_status == "not_applicable":
                bucket["LSF_not_applicable_count"] += 1
            elif lsf_status == "unscorable":
                bucket["LSF_unscorable_count"] += 1

        for dim in REPORT_DIMENSIONS:
            score = extract_dimension_score(eval_res, dim)
            if score is None:
                continue
            overall_bucket["scores"][dim].append(score)
            by_operation[operation]["scores"][dim].append(score)
            by_language[language]["scores"][dim].append(score)

    overall_avg = {}
    for dim in REPORT_DIMENSIONS:
        if overall_bucket["scores"][dim]:
            overall_avg[dim] = sum(overall_bucket["scores"][dim]) / len(overall_bucket["scores"][dim])

    op_stats = {}
    for op, bucket in by_operation.items():
        op_stats[op] = {
            'total': bucket['count'],
            'completed': bucket['count'],
            'LSF_valid_count': bucket['LSF_valid_count'],
            'LSF_not_applicable_count': bucket['LSF_not_applicable_count'],
            'LSF_unscorable_count': bucket['LSF_unscorable_count'],
        }
        for dim in REPORT_DIMENSIONS:
            if bucket['scores'][dim]:
                op_stats[op][f'{dim}_avg'] = sum(bucket['scores'][dim]) / len(bucket['scores'][dim])

    lang_stats = {}
    for lang, bucket in by_language.items():
        lang_stats[lang] = {
            'total': bucket['count'],
            'completed': bucket['count'],
            'LSF_valid_count': bucket['LSF_valid_count'],
            'LSF_not_applicable_count': bucket['LSF_not_applicable_count'],
            'LSF_unscorable_count': bucket['LSF_unscorable_count'],
        }
        for dim in REPORT_DIMENSIONS:
            if bucket['scores'][dim]:
                lang_stats[lang][f'{dim}_avg'] = sum(bucket['scores'][dim]) / len(bucket['scores'][dim])

    # Save statistics
    statistics = {
        'total_tasks': total,
        'completed': success_count,
        'failed': failed_count,
        'success_rate': success_count / total if total > 0 else 0,
        'LSF_valid_count': overall_bucket['LSF_valid_count'],
        'LSF_not_applicable_count': overall_bucket['LSF_not_applicable_count'],
        'LSF_unscorable_count': overall_bucket['LSF_unscorable_count'],
        'by_operation': op_stats,
        'by_language': lang_stats,
        'overall_averages': overall_avg
    }

    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    logger.info(f"Statistics saved to {stats_file}")

    # Generate text report
    report_lines = []
    report_lines.append("=== GPT-Image-1.5 Evaluation Results ===\n")
    report_lines.append("Overall Averages:")
    report_lines.append(f"{'Metric':<10} | {'Average':<10} | {'Valid Samples':<15}")
    report_lines.append("-" * 40)

    for dim in REPORT_DIMENSIONS:
        if dim in overall_avg:
            count = len([
                r for r in successful
                if extract_dimension_score(r.get('evaluation_results', {}), dim) is not None
            ])
            report_lines.append(f"{dim:<10} | {overall_avg[dim]:<10.2f} | {count:<15}")

    report_lines.append("")
    report_lines.append(
        f"LSF status counts: valid={overall_bucket['LSF_valid_count']}, "
        f"not_applicable={overall_bucket['LSF_not_applicable_count']}, "
        f"unscorable={overall_bucket['LSF_unscorable_count']}"
    )

    report_lines.append("\n\nBy Operation:")
    for op, stats in op_stats.items():
        report_lines.append(f"\n{op}:")
        report_lines.append(
            f"  LSF counts -> valid: {stats['LSF_valid_count']}, "
            f"not_applicable: {stats['LSF_not_applicable_count']}, "
            f"unscorable: {stats['LSF_unscorable_count']}"
        )
        for dim in REPORT_DIMENSIONS:
            key = f'{dim}_avg'
            if key in stats:
                report_lines.append(f"  {dim}: {stats[key]:.2f}")

    report_lines.append("\n\nBy Language:")
    for lang, stats in lang_stats.items():
        report_lines.append(f"\n{lang}:")
        report_lines.append(
            f"  LSF counts -> valid: {stats['LSF_valid_count']}, "
            f"not_applicable: {stats['LSF_not_applicable_count']}, "
            f"unscorable: {stats['LSF_unscorable_count']}"
        )
        for dim in REPORT_DIMENSIONS:
            key = f'{dim}_avg'
            if key in stats:
                report_lines.append(f"  {dim}: {stats[key]:.2f}")

    report_file = output_dir / 'averages.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report saved to {report_file}")

    # Print summary
    print("\n" + "="*60)
    print("Overall Averages:")
    for dim in REPORT_DIMENSIONS:
        if dim in overall_avg:
            print(f"  {dim}: {overall_avg[dim]:.2f}")
    print(
        f"LSF counts: valid={overall_bucket['LSF_valid_count']}, "
        f"not_applicable={overall_bucket['LSF_not_applicable_count']}, "
        f"unscorable={overall_bucket['LSF_unscorable_count']}"
    )
    print("="*60)

CATEGORY_SIZES = {"Art": 70, "Event": 60, "Fashion": 110, "Food": 30, "Quotes": 30}


def main():
    global RESPONSE_MODE
    parser = argparse.ArgumentParser(
        description="Semantic evaluation (LVM judge) for multilingual image text editing."
    )
    parser.add_argument("--category", type=str, required=True,
                        choices=sorted(CATEGORY_SIZES.keys()),
                        help="Dataset category (Art/Event/Fashion/Food/Quotes)")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Path to the dataset category directory (e.g. dataset/Quotes)")
    parser.add_argument("--pred_dir", type=Path, required=True,
                        help="Path to the model prediction category directory (e.g. predictions/<model>/Quotes)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent threads")
    parser.add_argument("--start_id", type=int, default=1,
                        help="Start ID (default: 1)")
    parser.add_argument("--end_id", type=int, default=None,
                        help="End ID (default: full size of the chosen category)")
    parser.add_argument("--languages", type=str, default=None,
                        help="Comma-separated language codes (default: all 12 languages)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--calculate_only", action="store_true",
                        help="Only calculate statistics from existing results")
    parser.add_argument("--input_file", type=Path, default=None,
                        help="Input file for calculate_only mode")
    parser.add_argument("--response_mode", type=str, choices=sorted(VALID_RESPONSE_MODES),
                        default=RESPONSE_MODE,
                        help="LLM response mode: nonstream, stream, or auto fallback")

    args = parser.parse_args()
    RESPONSE_MODE = args.response_mode
    logger.info(f"Response mode: {RESPONSE_MODE}")

    category = args.category
    if args.end_id is None:
        args.end_id = CATEGORY_SIZES[category]

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Enforce API key now (calculate_only mode does not need it)
    if not args.calculate_only and not API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Calculate only mode
    if args.calculate_only:
        input_file = args.input_file or (args.output_dir / f"{category}.json")
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)

        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        generate_statistics(results, args.output_dir)
        return

    # Validate paths
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not args.pred_dir.exists():
        logger.error(f"Prediction directory not found: {args.pred_dir}")
        sys.exit(1)

    # Parse languages
    languages = None
    if args.languages:
        languages = [l.strip() for l in args.languages.split(',')]
        logger.info(f"Filtering languages: {languages}")

    # Scan dataset
    logger.info("Scanning dataset...")
    all_tasks = scan_dataset(
        args.input_dir, args.pred_dir,
        start_id=args.start_id, end_id=args.end_id,
        languages=languages, category=category
    )

    logger.info(f"Found {len(all_tasks)} tasks")

    if not all_tasks:
        logger.error("No tasks found")
        sys.exit(1)

    # Check resume state
    output_file = args.output_dir / f"{category}.json"
    failed_file = args.output_dir / f"{category}_failed.json"

    if args.resume:
        all_tasks = check_resume_state(all_tasks, output_file)
        logger.info(f"Remaining tasks: {len(all_tasks)}")

    if not all_tasks:
        logger.info("No new tasks to evaluate")

        # Load existing results and generate statistics
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            generate_statistics(results, args.output_dir)
        return

    # Process tasks
    logger.info(f"Starting evaluation with {args.workers} workers...")

    # Thread-safe file writing lock
    file_lock = Lock()

    success_count = 0
    failure_count = 0

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="Evaluator") as executor:
        futures = {
            executor.submit(process_task_wrapper, task, args.input_dir, args.pred_dir, category): task
            for task in all_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            task = futures[future]
            try:
                result = future.result()

                if result['status'] == 'success':
                    # Immediately append to results file
                    append_result_to_file(result, output_file, file_lock)
                    success_count += 1
                else:
                    # Immediately append to failures file
                    append_result_to_file(result, failed_file, file_lock)
                    failure_count += 1

            except Exception as e:
                logger.error(f"Critical error processing task {task['task_id']}: {e}")
                error_result = {
                    'status': 'error',
                    'task_id': task['task_id'],
                    'error': str(e)
                }
                append_result_to_file(error_result, failed_file, file_lock)
                failure_count += 1

    logger.info(f"\nCompleted evaluation:")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")

    # Load final results and generate statistics
    final_results = []
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            final_results = json.load(f)

    generate_statistics(final_results, args.output_dir)

if __name__ == "__main__":
    main()

