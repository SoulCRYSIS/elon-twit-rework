"""Shared utilities."""

import re


def extract_bracket_range(question: str) -> str | None:
    """Extract bracket range from market question."""
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", question)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"less than (\d+)", question, re.I)
    if m:
        return f"0-{int(m.group(1))-1}"
    m = re.search(r"(\d+)\+", question)
    if m:
        return f"{m.group(1)}+"
    m = re.search(r"more than (\d+)", question, re.I)
    if m:
        return f"{int(m.group(1))+1}+"
    return None
