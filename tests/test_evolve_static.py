"""Tests for the evolution system's GPU-free parts: diff application and static rule checks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alphaevolve.checks import check_program  # noqa: E402
from alphaevolve.diff import DiffError, apply_response, change_set, change_similarity  # noqa: E402

PROGRAM = '''import os
# EVOLVE-BLOCK-START
_model_name = "Qwen/Qwen3-VL-2B-Instruct"
MAX_NEW_TOKENS = 64


def vlm_inference(image_path, question):
    return generate(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
# EVOLVE-BLOCK-END
print("outside")
'''


def diff(search, replace):
    return f"Some reasoning.\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_diff_replaces_inside_block():
    new, kind = apply_response(PROGRAM, diff("MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = 32"))
    assert kind == "diff" and "MAX_NEW_TOKENS = 32" in new and new.endswith('print("outside")\n')


def test_several_diffs_apply_in_order():
    response = diff("MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = 32") + diff("= 32", "= 16")
    assert "MAX_NEW_TOKENS = 16" in apply_response(PROGRAM, response)[0]


def test_diff_outside_block_is_rejected():
    with pytest.raises(DiffError, match="found 0 times"):
        apply_response(PROGRAM, diff("import os", "import sys"))


def test_ambiguous_diff_is_rejected():
    with pytest.raises(DiffError, match="found 2 times"):
        apply_response(PROGRAM, diff("MAX_NEW_TOKENS", "N"))


def test_trailing_whitespace_in_search_is_tolerated():
    new, _ = apply_response(PROGRAM, diff("MAX_NEW_TOKENS = 64   ", "MAX_NEW_TOKENS = 48"))
    assert "MAX_NEW_TOKENS = 48" in new


def test_full_rewrite_keeps_code_outside_block():
    rewrite = "```python\n# EVOLVE-BLOCK-START\ndef vlm_inference(image_path, question):\n    return 'x'\n# EVOLVE-BLOCK-END\n```"
    new, kind = apply_response(PROGRAM, rewrite)
    assert kind == "rewrite" and new.startswith("import os\n") and "return 'x'" in new and "outside" in new


def test_response_without_change_is_rejected():
    with pytest.raises(DiffError):
        apply_response(PROGRAM, "I would lower the token budget.")


def test_seed_programs_pass_checks():
    root = Path(__file__).resolve().parent.parent
    assert check_program((root / "manual_instruct.py").read_text(), model="Instruct") == []
    assert check_program((root / "manual_thinking.py").read_text(), model="Thinking") == []


@pytest.mark.parametrize("change, expected", [
    ("do_sample=False", None),
    ("do_sample=True", "non-greedy"),
    ("do_sample=False, temperature=0.7", "non-greedy"),
    ("do_sample=False, num_beams=4", "non-greedy"),
    ("do_sample=flag", "literal constant"),
])
def test_decoding_rules(change, expected):
    problems = check_program(PROGRAM.replace("do_sample=False", change))
    assert (problems == []) if expected is None else any(expected in p for p in problems)


@pytest.mark.parametrize("old, new, expected", [
    ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "model not allowed"),
    ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Thinking", "this run evolves"),
    ("import os", "import evaluate", "forbidden import"),
    ("import os", "DATA = 'charxiv/data/descriptive_val.json'", "forbidden path"),
    ("def vlm_inference(", "def other(", "no top-level function"),
    ("MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = = 64", "syntax error"),
])
def test_rule_violations(old, new, expected):
    assert any(expected in p for p in check_program(PROGRAM.replace(old, new), model="Instruct"))


def test_change_similarity_ignores_comments_and_layout():
    a = PROGRAM.replace("MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = 32  # shorter")
    b = PROGRAM.replace("MAX_NEW_TOKENS = 64", "\nMAX_NEW_TOKENS  =  32")
    c = PROGRAM.replace("do_sample=False)", "do_sample=False, use_cache=True)")
    assert change_similarity(change_set(PROGRAM, a), change_set(PROGRAM, b)) == 1.0
    assert change_similarity(change_set(PROGRAM, a), change_set(PROGRAM, c)) == 0.0
