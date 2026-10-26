"""Regression tests for manual_instruct.postprocess (no GPU; the model is loaded lazily and never here).

    python -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from manual_instruct import postprocess  # noqa: E402

Q = "question text is not used yet"


@pytest.mark.parametrize("response, expected", [
    # explicit answer markers
    ("There are 4 discrete labels in the legend.\n\nAnswer: 4", "4"),
    ("Thus, there are **3** lines in the plot.\n\n**Answer:** 3", "3"),
    ("We cannot determine it.\n\nTherefore, the answer is: Not Applicable.", "Not Applicable"),
    # bare final line after reasoning
    ("Therefore, the difference is 0.2.\n\n0.2", "0.2"),
    ("Thus, the layout is 2 by 2.\n\n2 by 2", "2 by 2"),
    # final sentence "... is X."
    ("The spatially highest labeled tick on the y-axis is 60.", "60"),
    ('The rightmost labeled tick on the x-axis for the subplot at row 1 and column 1 is "BiT-R101x3".', "BiT-R101x3"),
    ("Therefore, there is **1** line in the plot.", "1"),
    ("The tick marks are 0, 5. The lowest tick value on the y-axis is $10^{-5}$.", "10^-5"),
    ("The grid has four subplots. Thus, the layout is 1 by 4.", "1 by 4"),
])
def test_extracts_short_answer_from_prose(response, expected):
    assert postprocess(Q, response) == expected


@pytest.mark.parametrize("response", [
    # bare answers, including long ones, must pass through unchanged
    "60",
    "Not Applicable",
    "2 by 1",
    "Daily New Confirmed COVID-19 Case Number (Weekly Moving Average)",
    "advmod, amod, case, cc, conj, det, mark, nmod, nsubj, obj, obl",
    "Score = 0, Score = 1, Score = 2, Score = 3, Score = 4, Score = 5",
])
def test_bare_answers_unchanged(response):
    assert postprocess(Q, response) == response


@pytest.mark.parametrize("response, expected", [
    # unicode superscripts follow the gold answers' caret notation, in bare answers too
    ("PPL (10²)", "PPL (10^2)"),
    ("10⁻³", "10^-3"),
])
def test_superscripts_become_caret_notation(response, expected):
    assert postprocess(Q, response) == expected

