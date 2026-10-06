"""Turning an example into a model prompt for one of the two experimental conditions.

Both conditions use the same template; the only difference is whether the
surrounding sentences are included in the passage.
"""

import hashlib
from pathlib import Path

CONDITIONS = ("sentence", "context")


def mark_target(ex):
    """Target sentence with the expression wrapped in <expr> tags and the sentence in <target> tags."""
    sentence, span = ex["target_sentence"], ex["span"]
    marked = sentence.replace(span, f"<expr>{span}</expr>", 1)
    return f"<target>{marked}</target>"


def build_passage(ex, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    target = mark_target(ex)
    if condition == "sentence":
        return target
    parts = [ex.get("context_before", ""), target, ex.get("context_after", "")]
    return " ".join(p.strip() for p in parts if p and p.strip())


def load_template(path):
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template, ex, condition):
    return template.format(expression=ex["expression"], passage=build_passage(ex, condition))


def template_hash(template):
    """Short fingerprint saved with every prediction, so results can be traced to the exact prompt."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
