"""Shared utilities."""

import re


def progress_log(
    script_id: str,
    message: str,
    *,
    step: int | None = None,
    total: int | None = None,
) -> None:
    """Print one grep-friendly line: ``[script_id]`` or ``[script_id step/total]``."""
    if step is not None and total is not None and total > 0:
        prefix = f"[{script_id} {step}/{total}]"
    else:
        prefix = f"[{script_id}]"
    print(f"{prefix} {message}", flush=True)


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
