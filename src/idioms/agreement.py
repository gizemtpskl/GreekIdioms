"""Inter-annotator agreement, for examples that also have a second annotator's label.

    python -m idioms.agreement

Add a `label_reviewer2` field ("literal" or "idiomatic") to the examples the
second annotator labelled, without showing them the first label.
"""

from .data import SPLITS, read_jsonl
from .metrics import cohen_kappa


def main():
    examples = [ex for path in SPLITS.values() for ex in read_jsonl(path) if ex.get("label_reviewer2")]
    if not examples:
        raise SystemExit("No example has a 'label_reviewer2' yet, so agreement can't be computed.")
    first = [ex["label"] for ex in examples]
    second = [ex["label_reviewer2"] for ex in examples]
    agree = sum(a == b for a, b in zip(first, second))
    print(f"double-annotated examples: {len(examples)}")
    print(f"raw agreement: {agree / len(examples):.3f} ({agree}/{len(examples)})")
    print(f"Cohen's kappa: {cohen_kappa(first, second):.3f}")
    for ex in examples:
        if ex["label"] != ex["label_reviewer2"]:
            print(f"  disagree {ex['id']}: {ex['label']} vs {ex['label_reviewer2']} | {ex['target_sentence']}")


if __name__ == "__main__":
    main()
