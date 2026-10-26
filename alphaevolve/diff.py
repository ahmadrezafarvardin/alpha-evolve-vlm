"""Apply an LLM's proposed change to a program, restricted to its EVOLVE block.

The LLM answers with one or more SEARCH/REPLACE blocks (the AlphaEvolve diff format):

    <<<<<<< SEARCH
    exact lines from the current program
    =======
    replacement lines
    >>>>>>> REPLACE

or, as a fallback, with a full rewrite of the program in a ```python fenced block that keeps the
`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers. Code outside the block never changes.
"""
import difflib
import re

BLOCK_START, BLOCK_END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
_DIFF = re.compile(r"<{5,}\s*SEARCH\s*\n(.*?)\n?={5,}\s*\n(.*?)\n?>{5,}\s*REPLACE", re.DOTALL)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class DiffError(ValueError):
    """The response cannot be applied; the message is shown to the LLM-facing log and the event log."""


def split_block(code):
    """(before, block, after) around the EVOLVE block; the markers stay in `before` and `after`."""
    start, end = code.find(BLOCK_START), code.find(BLOCK_END)
    if start < 0 or end < start:
        raise DiffError("program has no EVOLVE-BLOCK-START/END markers")
    start += len(BLOCK_START)
    return code[:start], code[start:end], code[end:]


def parse_diffs(response):
    return [(search, replace) for search, replace in _DIFF.findall(response)]


def apply_diffs(code, diffs):
    before, block, after = split_block(code)
    for i, (search, replace) in enumerate(diffs, 1):
        if not search.strip():
            raise DiffError(f"diff {i}: empty SEARCH")
        count = block.count(search)
        if count == 0:
            # Tolerate trailing-whitespace differences, the most common LLM copy error.
            stripped = "\n".join(line.rstrip() for line in search.splitlines())
            block_stripped = "\n".join(line.rstrip() for line in block.splitlines())
            if block_stripped.count(stripped) == 1:
                block, search = block_stripped, stripped
                count = 1
        if count != 1:
            raise DiffError(f"diff {i}: SEARCH text found {count} times in the EVOLVE block (must be exactly once)")
        block = block.replace(search, replace, 1)
    return before + block + after


def apply_response(code, response):
    """New program from an LLM response, plus the kind of change ("diff" or "rewrite")."""
    diffs = parse_diffs(response)
    if diffs:
        return apply_diffs(code, diffs), "diff"
    for fenced in _FENCE.findall(response):
        if BLOCK_START in fenced and BLOCK_END in fenced:
            before, _, after = split_block(code)
            _, new_block, _ = split_block(fenced)
            return before + new_block + after, "rewrite"
    raise DiffError("response contains no SEARCH/REPLACE block and no full program with EVOLVE markers")


def _normalize_line(line):
    """Code line without comments and with collapsed whitespace ("" for blank or comment-only lines)."""
    line = line.strip()
    if line.startswith("#"):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"\s+#.*$", "", line))


def change_set(parent_code, child_code):
    """The edit from parent to child as a set of normalized "+line" / "-line" entries (comments and layout ignored)."""
    changes = set()
    for line in difflib.unified_diff(parent_code.splitlines(), child_code.splitlines(), lineterm="", n=0):
        if line[:1] in "+-" and not line.startswith(("+++", "---")):
            normalized = _normalize_line(line[1:])
            if normalized:
                changes.add(line[0] + normalized)
    return changes


def change_similarity(a, b):
    """Jaccard similarity of two change sets (1.0 = the same edit)."""
    return len(a & b) / len(a | b) if a | b else 1.0
