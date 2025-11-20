from dataclasses import dataclass
from typing import List


@dataclass
class SafetyResult:
    has_violation: bool
    reasons: List[str]


FORBIDDEN_PHRASES = [
    "kill yourself",
    "how to build a bomb",
]


def run_safety_checks(output_text: str) -> SafetyResult:
    """Very simple safety heuristics.

    - Flags obvious disallowed content based on phrase matching.
    - Can be extended with more advanced checks later.
    """
    reasons: List[str] = []

    lower = output_text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            reasons.append(f"Contains forbidden phrase: {phrase!r}")

    has_violation = len(reasons) > 0
    return SafetyResult(has_violation=has_violation, reasons=reasons)
