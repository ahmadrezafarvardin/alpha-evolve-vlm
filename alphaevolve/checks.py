"""Cascade stage 0: static checks that run before any GPU time is spent (CONTRACT.md §1).

A candidate is rejected if it does not compile, does not define `vlm_inference`, or breaks a rule:
greedy decoding only, only the two allowed models, and no access to gold answers or the evaluator.
"""
import ast
import re

ALLOWED_MODEL = re.compile(r"Qwen3-VL-2B-(Instruct|Thinking)\b")
# Sampling switches that make decoding non-greedy when set to these values.
_NON_GREEDY = {"do_sample": lambda v: v is True, "num_beams": lambda v: isinstance(v, int) and v > 1,
               "temperature": lambda v: isinstance(v, (int, float)) and v > 0,
               "top_p": lambda v: isinstance(v, (int, float)) and v < 1,
               "top_k": lambda v: isinstance(v, int) and v > 0}
_FORBIDDEN_STRINGS = ("charxiv/data", "descriptive_val", "reasoning_val", "answers.json", "scores/", "evaluate.py")
_FORBIDDEN_IMPORTS = ("evaluate", "descriptive_utils")


def check_program(code, model=None):
    """List of rule violations (empty if the program passes). `model` pins "Instruct" or "Thinking"."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    problems = []
    if not any(isinstance(n, ast.FunctionDef) and n.name == "vlm_inference" for n in tree.body):
        problems.append("no top-level function vlm_inference")

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if "Qwen/" in text:
                match = ALLOWED_MODEL.search(text)
                if not match:
                    problems.append(f"model not allowed: {text!r}")
                elif model and match.group(1) != model:
                    problems.append(f"this run evolves Qwen3-VL-2B-{model}, found {text!r}")
            for bad in _FORBIDDEN_STRINGS:
                if bad in text:
                    problems.append(f"forbidden path or file referenced: {bad!r}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
            for name in filter(None, names):
                if name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    problems.append(f"forbidden import: {name}")
        elif isinstance(node, ast.keyword) and node.arg in _NON_GREEDY:
            if isinstance(node.value, ast.Constant) and _NON_GREEDY[node.arg](node.value.value):
                problems.append(f"non-greedy decoding: {node.arg}={node.value.value!r}")
            elif not isinstance(node.value, ast.Constant) and node.arg in ("do_sample", "num_beams"):
                problems.append(f"{node.arg} must be a literal constant so greedy decoding can be verified")
    return sorted(set(problems))
