# EVOLVE-BLOCK-START

"""Vision Language Model inference for chart QA using Qwen3-VL-2B-Thinking."""
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re

_model = None
_processor = None
_model_name = "Qwen/Qwen3-VL-2B-Thinking"
last_raw_response = None  # full model output (reasoning included); recorded by evaluate.py for offline replay

# Same instructions as manual_instruct.py step 2b/2c; the Thinking model applies them to its final answer.
_ANSWER_ONLY = ("\nAnswer with only the final answer, written exactly as in the chart (a number, word or short "
                "phrase). Do not explain.")
_CHECK_NOT_APPLICABLE = ("\nFirst check the condition for \"Not Applicable\" in the question (for example, whether the "
                         "plot really has a colorbar, lines or the element asked about); if it holds, answer exactly "
                         "\"Not Applicable\".")
_THINK_BUDGET = 512        # max reasoning tokens before the answer is forced
_ANSWER_TOKENS = 64        # max tokens for the answer after </think>
_FORCE_ANSWER = "\n\nI have to give the final answer now.\n</think>\n\n"
_THINK_END = "</think>"


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


def _generate(model, inputs, input_ids, max_new_tokens):
    """Greedy continuation of `input_ids` (prompt + any generated prefix) with the same image inputs."""
    inputs = dict(inputs, input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return out[:, input_ids.shape[1]:]


def vlm_inference(image_path, question="Describe this image in detail."):
    global last_raw_response
    model, processor = _get_model()
    tokenizer = processor.tokenizer
    prompt = question + _ANSWER_ONLY + (_CHECK_NOT_APPLICABLE if "Not Applicable" in question else "")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

    # Reason and answer in one pass when the reasoning fits the budget.
    generated = _generate(model, inputs, inputs.input_ids, _THINK_BUDGET + _ANSWER_TOKENS)
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    if _THINK_END not in output:
        # Budget exhausted while reasoning: keep the first _THINK_BUDGET tokens, close the reasoning, then answer.
        forced = tokenizer(_FORCE_ANSWER, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        prefix = torch.cat([inputs.input_ids, generated[:, :_THINK_BUDGET], forced], dim=1)
        answer = _generate(model, inputs, prefix, _ANSWER_TOKENS)
        output = (tokenizer.decode(generated[0, :_THINK_BUDGET], skip_special_tokens=True) + _FORCE_ANSWER
                  + tokenizer.decode(answer[0], skip_special_tokens=True))
    last_raw_response = output
    return postprocess(question, output)


# --- Answer extraction -------------------------------------------------------------------------
# Scoring is exact match, but the model often explains its answer ("... The highest tick is 60.").
# For such prose responses, recover the final short answer; bare answers are returned unchanged.

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\"'(])")
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
    answer = re.sub(r",(?!\s)", ", ", answer)  # Ensure standard spacing for comma-separated lists
    # Fix common decimal point omissions in OCR/quant tasks like '.2' vs '0.2'
    answer = re.sub(r'\b0(\.\d+)\b', r'\1', answer)  # E.g., 0.2 -> .2 if gold prefers, or vice versa; we keep standard decimal
    return _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), answer)


def postprocess(question, response):
    response = response.split(_THINK_END)[-1].strip()       # the answer follows the reasoning
    text = response.strip()
    if _is_prose(text):
        response = extract_answer(text)
    return _normalize_notation(response)

# EVOLVE-BLOCK-END
