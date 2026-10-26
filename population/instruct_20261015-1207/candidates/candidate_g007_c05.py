# EVOLVE-BLOCK-START
"""Vision Language Model inference for chart QA using Qwen3-VL-2B-Instruct or Thinking."""
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re

_model = None
_processor = None
_model_name = "Qwen/Qwen3-VL-2B-Instruct" # or Qwen/Qwen3-VL-2B-Thinking
last_raw_response = None  # model output before postprocess(); recorded by evaluate.py for offline replay

# Step 2b: ask for the bare answer. Explanations were 82-88 % of the runtime and rarely correct.
_ANSWER_ONLY = ("\nAnswer with only the final exact value, number, label, or phrase. Do not explain.")
_MAX_NEW_TOKENS = 64  # longest dev gold answer is 30 tokens
# Step 2c: the model rarely uses the questions' "Not Applicable" option (recall 20 %), e.g. it reads axis ticks
# as a colorbar or hallucinates elements. Ask it to check that condition carefully first.
_CHECK_NOT_APPLICABLE = ("\nFirst check carefully whether the requested element (such as colorbar, continuous legend, lines, intersection, or specific axis) actually exists in the specified subplot. If it does not exist or the condition for \"Not Applicable\" holds, you must answer exactly \"Not Applicable\".")


def _get_model():
    global _model, _processor
    if _model is None:
        _model = AutoModelForVision2Seq.from_pretrained(
            _model_name, torch_dtype=torch.float16, device_map="auto",
        )
        _model.eval()
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(_model_name, use_fast=False)
    return _model, _processor


def vlm_inference(image_path, question="Describe this image in detail."):
    model, processor = _get_model()
    messages = [
        {"role": "system", "content": "You are a precise chart-reading assistant. Read the chart carefully and output only the exact final answer."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question + _ANSWER_ONLY
                                     + (_CHECK_NOT_APPLICABLE if "Not Applicable" in question else "")},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    global last_raw_response
    last_raw_response = output_text[0]
    return postprocess(question, output_text[0])


# --- Answer extraction -------------------------------------------------------------------------
# Scoring is exact match, but the model often explains its answer ("... The highest tick is 60.").
# For such prose responses, recover the final short answer; bare answers are returned unchanged.

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[*\"'(])")
_ANSWER_MARKER = re.compile(r"\banswer\b\W*?(?:\bis\b)?\s*[:：]\s*(.+)$", re.IGNORECASE)
_ANSWER_IS = re.compile(r"\bthe (?:final )?answer is\s+(.+)$", re.IGNORECASE)
_LAST_IS = re.compile(r".*\b(?:is|are|equals)\s+(.+)$", re.IGNORECASE)
_LEADING_COUNT = re.compile(r"^(-?\d[\d,.]*)\s+(?!by\b)[a-z]")    # not the "n by m" layout format
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")


def _is_prose(text):
    """A multi-line or sentence-like response, as opposed to a bare answer such as a label or a list."""
    return ("\n" in text or bool(_SENTENCE_BREAK.search(text))
            or (text.endswith(".") and len(text.split()) > 6))


def _clean(answer):
    answer = answer.strip().replace("**", "")
    answer = re.sub(r"^\$\$?(.*?)\$\$?$", r"\1", answer)      # strip display/inline math wrappers
    answer = re.sub(r"\$([^$]*)\$", r"\1", answer)            # $10^{-5}$ -> 10^{-5}
    answer = re.sub(r"\^\{([^}]*)\}", r"^\1", answer)          # 10^{-5} -> 10^-5
    answer = answer.rstrip(".:;").strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'`":
        answer = answer[1:-1].strip()
    count = _LEADING_COUNT.match(answer)                       # "4 discrete labels in the legend" -> "4"
    return count.group(1).rstrip(".,") if count else answer


def _is_bare(line):
    """A final line that is just an answer: short, not a bullet, heading, sentence or calculation."""
    return (len(line) <= 40 and len(line.split()) <= 6 and not line.startswith(("-", "*", "#", "|"))
            and not re.search(r"[:=]|\.$", line.replace("...", "")))


def extract_answer(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
      return text
    for line in reversed(lines):                               # "Answer: 4", "**Answer:** 3"
        match = _ANSWER_MARKER.search(line) or _ANSWER_IS.search(line)
        if match and _clean(match.group(1)):
            return _clean(match.group(1))
    if len(lines) > 1 and _is_bare(lines[-1]):                 # "...is 0.2.\n\n0.2"
        return _clean(lines[-1])
    last_sentence = _SENTENCE_BREAK.split(lines[-1])[-1]       # "...on the y-axis is 60."
    match = _LAST_IS.match(last_sentence)
    if match and _clean(match.group(1)):
        return _clean(match.group(1))
    return text


def _normalize_notation(answer):
    """Unicode superscripts as in the gold answers' caret notation: 10² -> 10^2, 10⁻³ -> 10^-3."""
    return _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), answer)


def postprocess(question, response):
    text = response.strip()
    if _is_prose(text):
        response = extract_answer(text)
    else:
        response = _clean(response)
    return _normalize_notation(response)

# EVOLVE-BLOCK-END
