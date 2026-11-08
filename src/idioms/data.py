"""The dataset: data/pilot/pilot.jsonl is the development split, data/final/final.jsonl the test split.

    python -m idioms.data        # check both files and print an overview
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"dev": ROOT / "data/pilot/pilot.jsonl", "test": ROOT / "data/final/final.jsonl"}
LABELS = ("literal", "idiomatic")
ANNOTATION_LABELS = LABELS + ("ambiguous",)

# field -> expected type; every example must have all of them
FIELDS = {
    "id": str,
    "expression_id": str,
    "expression": str,
    "target_sentence": str,
    "span": str,
    "context_before": str,
    "context_after": str,
    "label": str,
    "sentence_sufficient": bool,
    "source": str,
    "is_constructed": bool,
}


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise SystemExit(f"{path}, line {line_no}: expected a JSON object")
                    records.append(record)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}, line {line_no}: not valid JSON ({e})") from None
    return records


def write_jsonl(path, records, mode="w"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check(examples):
    """Problems in a list of examples, as readable strings (empty list = all good)."""
    problems = []
    seen = set()
    for ex in examples:
        where = ex.get("id", "?")
        for field, kind in FIELDS.items():
            if not isinstance(ex.get(field), kind):
                problems.append(f"{where}: '{field}' should be a {kind.__name__}, got {ex.get(field)!r}")
            elif kind is str and field not in ("context_before", "context_after") and not ex[field].strip():
                problems.append(f"{where}: '{field}' must not be empty")
        if isinstance(ex.get("id"), str):
            if ex["id"] in seen:
                problems.append(f"{where}: duplicate id")
            seen.add(ex["id"])
        if ex.get("label") not in ANNOTATION_LABELS:
            problems.append(f"{where}: label must be one of {ANNOTATION_LABELS}, got {ex.get('label')!r}")
        if (isinstance(ex.get("span"), str) and isinstance(ex.get("target_sentence"), str)
                and ex["span"] not in ex["target_sentence"]):
            problems.append(f"{where}: span {ex.get('span')!r} is not in the target sentence")
        problems.extend(check_review(ex))
    return problems


def check_review(ex):
    """Validate independent judgments and their relationship to the evaluation label."""
    fields = ("label_reviewer1", "label_reviewer2", "sentence_sufficient_reviewer1",
              "sentence_sufficient_reviewer2", "reviewer2", "notes_reviewer2",
              "review_status", "adjudicator", "adjudication_notes")
    if not any(field in ex for field in fields):
        return []
    problems = []
    where = ex.get("id", "?")
    status = ex.get("review_status")
    if status not in ("pending", "agreed", "adjudicated"):
        problems.append(f"{where}: invalid or missing review_status")
    for field in ("label_reviewer1", "label_reviewer2"):
        if ex.get(field) not in ANNOTATION_LABELS:
            problems.append(f"{where}: invalid or missing {field}")
    for field in ("sentence_sufficient_reviewer1", "sentence_sufficient_reviewer2"):
        if not isinstance(ex.get(field), bool):
            problems.append(f"{where}: {field} must be true or false")
    for field in ("annotator", "reviewer2"):
        if not isinstance(ex.get(field), str) or not ex[field].strip():
            problems.append(f"{where}: reviewed example requires {field}")
    if (isinstance(ex.get("annotator"), str) and isinstance(ex.get("reviewer2"), str)
            and ex["annotator"].strip() == ex["reviewer2"].strip()):
        problems.append(f"{where}: the second reviewer must be a different person")
    notes = ex.get("notes_reviewer2")
    if not isinstance(notes, str) or (ex.get("label_reviewer2") == "ambiguous" and not notes.strip()):
        problems.append(f"{where}: notes_reviewer2 must be text and explain an ambiguous label")
    disagrees = (ex.get("label_reviewer1") != ex.get("label_reviewer2") or
                 ex.get("sentence_sufficient_reviewer1") != ex.get("sentence_sufficient_reviewer2"))
    if status == "agreed" and disagrees:
        problems.append(f"{where}: agreed review contains a disagreement")
    if status == "pending" and not disagrees:
        problems.append(f"{where}: pending review requires a disagreement")
    if status in ("pending", "agreed"):
        if (ex.get("label") != ex.get("label_reviewer1") or
                ex.get("sentence_sufficient") != ex.get("sentence_sufficient_reviewer1")):
            problems.append(f"{where}: evaluation fields changed before adjudication")
    if status == "adjudicated":
        for field in ("adjudicator", "adjudication_notes"):
            if not isinstance(ex.get(field), str) or not ex[field].strip():
                problems.append(f"{where}: adjudication requires {field}")
    return problems


def load_split(split):
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {list(SPLITS)}, got {split!r}")
    datasets = {name: read_jsonl(path) for name, path in SPLITS.items()}
    for name, rows in datasets.items():
        problems = check(rows)
        if problems:
            raise SystemExit(f"{SPLITS[name]} has {len(problems)} problems:\n" + "\n".join(problems[:20]))
    for field in ("id", "expression_id", "expression"):
        shared = {ex[field] for ex in datasets["dev"]} & {ex[field] for ex in datasets["test"]}
        if shared:
            raise SystemExit(f"development/test overlap in {field}: {sorted(shared)}")
    examples = datasets[split]
    pending = [ex["id"] for ex in examples if ex.get("review_status") == "pending"]
    if pending:
        raise SystemExit(f"resolve pending annotation disagreements before inference: {pending}")
    return [ex for ex in examples if ex["label"] in LABELS]


def minimal_pairs(examples):
    """Pairs of ids with the same expression and target sentence but different labels:
    only the context tells them apart."""
    groups = defaultdict(list)
    for ex in examples:
        groups[(ex["expression_id"], ex["target_sentence"])].append(ex)
    pairs = []
    for group in groups.values():
        literal = [ex["id"] for ex in group if ex["label"] == "literal"]
        idiomatic = [ex["id"] for ex in group if ex["label"] == "idiomatic"]
        pairs += [(a, b) for a in literal for b in idiomatic]
    return sorted(pairs)


def overview(split, examples):
    labels = Counter(ex["label"] for ex in examples)
    sufficient = Counter(ex["sentence_sufficient"] for ex in examples)
    print(f"{split}: {SPLITS[split].relative_to(ROOT)}")
    print(f"  {len(examples)} examples, {len({ex['expression_id'] for ex in examples})} expressions, "
          f"{labels['literal']} literal / {labels['idiomatic']} idiomatic, "
          f"{sufficient[True]} sentence-sufficient / {sufficient[False]} need context, "
          f"{len(minimal_pairs(examples))} minimal pairs, "
          f"{sum(ex['is_constructed'] for ex in examples)} constructed")
    by_expression = defaultdict(Counter)
    names = {}
    for ex in examples:
        by_expression[ex["expression_id"]][ex["label"]] += 1
        names[ex["expression_id"]] = ex["expression"]
    for expr_id, counts in sorted(by_expression.items()):
        print(f"    {expr_id}  {names[expr_id]:42} literal {counts['literal']}  idiomatic {counts['idiomatic']}")


def main():
    for split in SPLITS:
        selected = load_split(split)
        overview(split, selected)
        print(f"  {len(read_jsonl(SPLITS[split])) - len(selected)} ambiguous examples excluded from evaluation")
        print()
    dev, test = load_split("dev"), load_split("test")
    shared = ({ex["expression"] for ex in dev} & {ex["expression"] for ex in test}) | \
             ({ex["expression_id"] for ex in dev} & {ex["expression_id"] for ex in test})
    print("No expression appears in both splits." if not shared else f"WARNING: in both splits: {sorted(shared)}")


if __name__ == "__main__":
    main()
