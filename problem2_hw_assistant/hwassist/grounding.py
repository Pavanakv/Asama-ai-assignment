"""Code-level hallucination check.

A system prompt asks the model to stay grounded; this *verifies* it. Every
number in the final answer must appear in (or be a trivial rounding/count of)
the tool results gathered this turn, or in the user's own question. Anything
else is reported so the agent can ask the model to retry, or warn the user.

This is deliberately simple (numbers only). It won't catch a wrong *word*
("DDR5" when the data says DDR4 - though that has a digit, so it does get
caught), but it blocks the most damaging failure for this product: invented or
mis-converted specs.
"""
from __future__ import annotations

import json
import re

_ANSWER_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")
_DATA_NUM = re.compile(r"\d+(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)


def _collect(obj, nums: set) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, nums)
    elif isinstance(obj, list):
        nums.add(float(len(obj)))  # "2 drives" is grounded by a 2-element list
        for v in obj:
            _collect(v, nums)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        nums.add(float(obj))
    elif isinstance(obj, str):
        nums.update(float(m) for m in _DATA_NUM.findall(obj))


def _grounded(x: float, allowed: set) -> bool:
    for a in allowed:
        if abs(a - x) < 1e-9 or round(a) == x or round(a, 1) == x or round(a, 2) == x:
            return True
    return False


def ungrounded_numbers(answer: str, tool_results: list, user_text: str = "") -> list:
    allowed: set = set()
    for r in tool_results:
        _collect(r, allowed)
    allowed.update(float(m) for m in _DATA_NUM.findall(user_text))
    body = _LIST_MARKER.sub("", answer)  # "1. foo" list numbering isn't a spec
    bad = []
    for m in _ANSWER_NUM.findall(body):
        if not _grounded(float(m), allowed) and m not in bad:
            bad.append(m)
    return bad


def results_digest(tool_results: list) -> str:
    return json.dumps(tool_results, ensure_ascii=False)
