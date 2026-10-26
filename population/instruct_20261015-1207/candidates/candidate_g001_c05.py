# EVOLVE-BLOCK-START
def _normalize_notation(answer):
    """Unicode superscripts as in the gold answers' caret notation: 10² -> 10^2, 10⁻³ -> 10^-3."""
    # Also fix common OCR / model spelling inconsistencies found in chart descriptions
    answer = answer.replace("stablizes", "stabilizes")
    return _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPT_RUN), answer)
# EVOLVE-BLOCK-END
