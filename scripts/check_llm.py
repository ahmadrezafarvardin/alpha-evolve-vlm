"""Check the mutation-LLM setup: key present, models listed by the gateway, one test call.

    .venv/bin/python scripts/check_llm.py                      # list Gemini models, call the default one
    .venv/bin/python scripts/check_llm.py --model <name>       # test a specific model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alphaevolve.llm import LLM  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Gemini-3.1-Flash-Lite-Preview")
    args = parser.parse_args()

    llm = LLM()  # fails with a clear message if ARVAN_API_KEY is missing
    names = sorted(m.id for m in llm.client.models.list())
    print(f"{len(names)} models; Gemini: {', '.join(n for n in names if 'gemini' in n.lower())}")
    text, record = llm.generate("Reply with one word.", "Say OK.", {"model": args.model}, temperature=0.0)
    print(f"{args.model}: {text.strip()!r} in {record['latency']:.1f}s, "
          f"{record['input_tokens']} in / {record['output_tokens']} out tokens")


if __name__ == "__main__":
    main()
