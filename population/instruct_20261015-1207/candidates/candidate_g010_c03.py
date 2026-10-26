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
_ANSWER_ONLY = ("\nAnswer with only the final answer, written exactly as in the chart (a number, word or short "
                "phrase). Do not explain.")
_MAX_NEW_TOKENS = 64  # longest dev gold answer is 30 tokens
# Step 2c: the model rarely uses the questions' "Not Applicable" option (recall 20 %), e.g. it reads axis ticks
# as a colorbar. Ask it to check that condition first, only for questions that have one.
_CHECK_NOT_APPLICABLE = ("\nFirst check the condition for \"Not Applicable\" in the question (for example, whether the "
                         "plot really has a colorbar, lines or the element asked about); if it holds, answer exactly "
                         "\"Not Applicable\".")

_image_cache = {}


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
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question + _ANSWER_ONLY
                                 + (_CHECK_NOT_APPLICABLE if "Not Applicable" in question else "")},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_path in _image_cache:
        image_inputs, video_inputs = _image_cache[image_path]
    else:
        image_inputs, video_inputs = process_vision_info(messages)
        _image_cache[image_path] = (image_inputs, video_inputs)
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
# EVOLVE-BLOCK-END
