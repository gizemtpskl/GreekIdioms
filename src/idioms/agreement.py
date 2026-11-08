"""Inter-annotator agreement on the subset labelled by a second reviewer.

    python -m idioms.agreement data/review/pilot/reviewed.jsonl
    python -m idioms.agreement  # both canonical splits

Compares preserved `label_reviewer1` and `label_reviewer2`, including ambiguous
labels. Never compares against adjudicated gold labels.
"""

import argparse
import math

from .data import ANNOTATION_LABELS, SPLITS, check, read_jsonl
from .metrics import cohen_kappa


def agreement_summary(records):
    reviewed = [ex for ex in records if ex.get("label_reviewer2")]
    if not reviewed:
        raise ValueError("no examples have a 'label_reviewer2' field yet")
    for ex in reviewed:
        if ex.get("label_reviewer1") not in ANNOTATION_LABELS:
            raise ValueError(f"{ex['id']}: missing or invalid original label_reviewer1; "
                             "use the review import workflow to preserve independent labels")
        if ex["label_reviewer2"] not in ANNOTATION_LABELS:
            raise ValueError(f"{ex['id']}: invalid label_reviewer2")
    problems = check(records)
    if problems:
        raise ValueError("invalid dataset:\n" + "\n".join(problems))
    a = [ex["label_reviewer1"] for ex in reviewed]
    b = [ex["label_reviewer2"] for ex in reviewed]
    return {
        "total": len(records), "n": len(reviewed),
        "agree": sum(x == y for x, y in zip(a, b)),
        "kappa": cohen_kappa(a, b),
        "disagreements": [ex for ex in reviewed if ex["label_reviewer1"] != ex["label_reviewer2"]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="*", help="reviewed JSONL files (default: both canonical splits)")
    args = ap.parse_args(argv)

    try:
        paths = args.data or SPLITS.values()
        summary = agreement_summary([ex for path in paths for ex in read_jsonl(path)])
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    n, agree = summary["n"], summary["agree"]
    print(f"double-annotated: {n}/{summary['total']} (including ambiguous labels)")
    print(f"raw agreement:    {agree / n:.3f} ({agree}/{n})")
    kappa = summary["kappa"]
    print("Cohen's kappa:    " + ("undefined (both reviewers used one identical category)"
                                  if math.isnan(kappa) else f"{kappa:.3f}"))
    for ex in summary["disagreements"]:
        print(f"  {ex['id']}: {ex['label_reviewer1']} vs {ex['label_reviewer2']}  | {ex['target_sentence']}")


if __name__ == "__main__":
    main()
