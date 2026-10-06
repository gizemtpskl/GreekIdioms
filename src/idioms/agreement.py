"""Inter-annotator agreement on the subset labelled by a second reviewer.

    python -m idioms.agreement data/pilot/examples.jsonl

Uses examples that have a `label_reviewer2` field; compares it with `label`.
"""

import argparse

from .data import load_jsonl
from .metrics import cohen_kappa


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    args = ap.parse_args(argv)

    pairs = [(ex["label"], ex["label_reviewer2"]) for ex in load_jsonl(args.data) if ex.get("label_reviewer2")]
    if not pairs:
        raise SystemExit("no examples have a 'label_reviewer2' field yet")
    a, b = zip(*pairs)
    agree = sum(x == y for x, y in pairs)
    print(f"double-annotated: {len(pairs)}")
    print(f"raw agreement:    {agree / len(pairs):.3f} ({agree}/{len(pairs)})")
    print(f"Cohen's kappa:    {cohen_kappa(list(a), list(b)):.3f}")
    disagreements = [ex for ex in load_jsonl(args.data) if ex.get("label_reviewer2") and ex["label"] != ex["label_reviewer2"]]
    for ex in disagreements:
        print(f"  {ex['id']}: {ex['label']} vs {ex['label_reviewer2']}  | {ex['target_sentence']}")


if __name__ == "__main__":
    main()
