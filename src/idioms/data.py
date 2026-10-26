"""The dataset: data/pilot/pilot.jsonl is the development split, data/final/final.jsonl the test split.

    python -m idioms.data        # check both files and print an overview
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"dev": ROOT / "data/pilot/pilot.jsonl", "test": ROOT / "data/final/final.jsonl"}
LABELS = ("literal", "idiomatic")

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
                    records.append(json.loads(line))
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
        if ex.get("id") in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(ex.get("id"))
        if ex.get("label") not in LABELS:
            problems.append(f"{where}: label must be 'literal' or 'idiomatic', got {ex.get('label')!r}")
        if isinstance(ex.get("span"), str) and ex.get("span") not in ex.get("target_sentence", ""):
            problems.append(f"{where}: span {ex.get('span')!r} is not in the target sentence")
    return problems


def load_split(split):
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {list(SPLITS)}, got {split!r}")
    examples = read_jsonl(SPLITS[split])
    problems = check(examples)
    if problems:
        raise SystemExit(f"{SPLITS[split]} has {len(problems)} problems:\n" + "\n".join(problems[:20]))
    return examples


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
        overview(split, load_split(split))
        print()
    dev, test = load_split("dev"), load_split("test")
    shared = ({ex["expression"] for ex in dev} & {ex["expression"] for ex in test}) | \
             ({ex["expression_id"] for ex in dev} & {ex["expression_id"] for ex in test})
    print("No expression appears in both splits." if not shared else f"WARNING: in both splits: {sorted(shared)}")


if __name__ == "__main__":
    main()
