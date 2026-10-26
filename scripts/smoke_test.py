"""Load a Qwen3-VL model and answer one CharXiv dev question with greedy decoding.

    python scripts/smoke_test.py                                   # Instruct, first dev query
    python scripts/smoke_test.py --model Qwen/Qwen3-VL-2B-Thinking --max-new-tokens 2048
"""
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import load_charxiv_data  # noqa: E402


def pick_dtype(name):
    if name == "auto":
        # T4 (sm_75) has no native bf16; emulating it is slow, so fall back to fp16.
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
    parser.add_argument("--index", type=int, default=0, help="which dev query to ask")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible: switch the Colab runtime to a GPU.")
    dtype = pick_dtype(args.dtype)
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype} | model: {args.model}")

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 1e9:.2f} GB on GPU")

    queries, data = load_charxiv_data(args.index + 1)
    key, query = list(queries.items())[args.index]
    gold = data[str(query["figure_id"])]["answers"][query["subq_idx"]]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": query["figure_path"]},
        {"type": "text", "text": query["question"]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print(f"\nQuery {key} (qid {query['qid']}), image {query['figure_path']}")
    print(f"Question: {query['question'].splitlines()[0]}")
    print(f"Answer:   {answer!r}")
    print(f"Gold:     {gold!r}  -> {'CORRECT' if answer.strip().lower() == gold.lower() else 'wrong'}")
    print(f"Generated {out.shape[1] - inputs.input_ids.shape[1]} tokens in {time.time() - t0:.1f}s (greedy)")


if __name__ == "__main__":
    main()
