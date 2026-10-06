"""Check an annotated data file and show the label balance per expression.

    python -m idioms.validate data/pilot/examples.jsonl [--splits data/pilot/splits.json]
"""

import argparse
from collections import Counter

from .data import label_counts_by_expression, load_jsonl, load_splits, validate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--splits")
    args = ap.parse_args(argv)

    examples = load_jsonl(args.data)
    problems = validate(examples)
    for p in problems:
        print("PROBLEM:", p)

    split_of = {}
    if args.splits:
        for name, expr_ids in load_splits(args.splits).items():
            split_of.update({e: name for e in expr_ids})

    print(f"\n{'expression_id':<15}{'split':<8}{'literal':>8}{'idiomatic':>10}{'ambiguous':>10}")
    for expr_id, counts in sorted(label_counts_by_expression(examples).items()):
        print(f"{expr_id:<15}{split_of.get(expr_id, '-'):<8}{counts['literal']:>8}{counts['idiomatic']:>10}{counts['ambiguous']:>10}")

    totals = Counter(ex["label"] for ex in examples)
    constructed = sum(ex.get("is_constructed") is True for ex in examples)
    print(f"\ntotal: {len(examples)} examples, {dict(totals)}, {constructed} constructed")
    if args.splits:
        missing = {ex["expression_id"] for ex in examples} - split_of.keys()
        if missing:
            print(f"not assigned to any split: {sorted(missing)}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
