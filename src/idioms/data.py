"""Loading, validating and splitting the annotated examples (JSONL, one example per line)."""

import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("literal", "idiomatic")  # the two classes used in the evaluation
ANNOTATION_LABELS = LABELS + ("ambiguous",)  # "ambiguous" is recorded but excluded from evaluation

REQUIRED_FIELDS = (
    "id",
    "expression_id",
    "expression",
    "target_sentence",
    "span",
    "label",
    "source",
    "is_constructed",
)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
    return records


def validate(examples):
    """Return a list of human-readable problems; an empty list means the file is OK."""
    problems = []
    seen_ids = set()
    for i, ex in enumerate(examples, 1):
        where = f"line {i} (id={ex.get('id')!r})"
        for field in REQUIRED_FIELDS:
            if field not in ex or ex[field] in (None, ""):
                problems.append(f"{where}: missing field '{field}'")
        if ex.get("id") in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(ex.get("id"))
        if ex.get("label") not in ANNOTATION_LABELS:
            problems.append(f"{where}: label must be one of {ANNOTATION_LABELS}, got {ex.get('label')!r}")
        span, sentence = ex.get("span"), ex.get("target_sentence")
        if span and sentence and span not in sentence:
            problems.append(f"{where}: span {span!r} does not occur in target_sentence")
        if not isinstance(ex.get("is_constructed"), bool):
            problems.append(f"{where}: is_constructed must be true or false")
    return problems


def label_counts_by_expression(examples):
    counts = defaultdict(Counter)
    for ex in examples:
        counts[ex["expression_id"]][ex["label"]] += 1
    return counts


def load_splits(path):
    """splits.json maps split name -> list of expression_ids, e.g. {"dev": [...], "test": [...]}."""
    with open(path, encoding="utf-8") as f:
        splits = json.load(f)
    seen = {}
    for name, expr_ids in splits.items():
        for expr_id in expr_ids:
            if expr_id in seen:
                raise ValueError(f"expression {expr_id!r} is in both '{seen[expr_id]}' and '{name}'")
            seen[expr_id] = name
    return splits


def select(examples, splits=None, split=None):
    """Keep binary-labelled examples, restricted to the expressions of `split` if given."""
    if split is not None:
        if splits is None or split not in splits:
            raise ValueError(f"unknown split {split!r}")
        allowed = set(splits[split])
        examples = [ex for ex in examples if ex["expression_id"] in allowed]
    return [ex for ex in examples if ex["label"] in LABELS]


def write_jsonl(path, records, mode="w"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
