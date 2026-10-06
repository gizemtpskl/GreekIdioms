"""Score prediction files, and compare two runs on the same items.

    # metrics table for every run (also written to results/metrics.csv)
    python -m idioms.evaluate results/predictions/*.jsonl

    # add per-expression and sentence-sufficient breakdowns
    python -m idioms.evaluate results/predictions/test__*.jsonl --breakdown

    # sentence-only vs. with-context for one model (McNemar + flip table)
    python -m idioms.evaluate --compare A_sentence.jsonl B_context.jsonl
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .data import load_jsonl
from .metrics import accuracy, bootstrap_ci, macro_f1, mcnemar_exact, paired_bootstrap_diff


def score(records):
    gold = [r["gold"] for r in records]
    pred = [r["prediction"] for r in records]
    acc_lo, acc_hi = bootstrap_ci(gold, pred, accuracy)
    f1_lo, f1_hi = bootstrap_ci(gold, pred, macro_f1)
    return {
        "n": len(records),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci": f"[{acc_lo:.2f}, {acc_hi:.2f}]",
        "macro_f1": macro_f1(gold, pred),
        "macro_f1_ci": f"[{f1_lo:.2f}, {f1_hi:.2f}]",
        "unparsed_or_refused": sum(r["prediction"] is None for r in records),
    }


def print_table(rows, columns):
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in columns))


def fmt(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def breakdown(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key)].append(r)
    rows = []
    for value, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        gold = [r["gold"] for r in group]
        rows.append({
            key: value,
            "n": len(group),
            "literal": gold.count("literal"),
            "idiomatic": gold.count("idiomatic"),
            "accuracy": accuracy(gold, [r["prediction"] for r in group]),
        })
    return rows


def compare(path_a, path_b):
    a = {r["id"]: r for r in load_jsonl(path_a)}
    b = {r["id"]: r for r in load_jsonl(path_b)}
    ids = sorted(a.keys() & b.keys())
    if len(ids) < max(len(a), len(b)):
        print(f"note: comparing only the {len(ids)} items present in both files")
    gold = [a[i]["gold"] for i in ids]
    pred_a = [a[i]["prediction"] for i in ids]
    pred_b = [b[i]["prediction"] for i in ids]
    correct_a = [g == p for g, p in zip(gold, pred_a)]
    correct_b = [g == p for g, p in zip(gold, pred_b)]
    both = sum(x and y for x, y in zip(correct_a, correct_b))
    only_a = sum(x and not y for x, y in zip(correct_a, correct_b))
    only_b = sum(y and not x for x, y in zip(correct_a, correct_b))
    neither = len(ids) - both - only_a - only_b

    print(f"A = {path_a}\nB = {path_b}\n")
    print(f"accuracy A = {accuracy(gold, pred_a):.3f}   B = {accuracy(gold, pred_b):.3f}")
    lo, hi = paired_bootstrap_diff(gold, pred_a, pred_b, accuracy)
    print(f"accuracy B - A = {accuracy(gold, pred_b) - accuracy(gold, pred_a):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"\n                 B correct   B wrong")
    print(f"A correct        {both:9d}   {only_a:7d}")
    print(f"A wrong          {only_b:9d}   {neither:7d}")
    print(f"\nB fixed {only_b} of A's errors and introduced {only_a} new ones.")
    print(f"McNemar exact p = {mcnemar_exact(only_a, only_b):.4f}")

    flips = [(i, a[i], b[i]) for i in ids if (a[i]["gold"] == a[i]["prediction"]) != (b[i]["gold"] == b[i]["prediction"])]
    if flips:
        print("\nItems where the two runs disagree in correctness:")
        for i, ra, rb in flips:
            print(f"  {i} ({ra['expression_id']}): gold={ra['gold']}  A={ra['prediction']}  B={rb['prediction']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="prediction files to score")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="paired comparison of two runs")
    ap.add_argument("--breakdown", action="store_true", help="per-expression and sentence-sufficient tables")
    ap.add_argument("--csv", default="results/metrics.csv")
    args = ap.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return
    if not args.files:
        ap.error("give prediction files to score, or --compare A B")

    rows = []
    for path in args.files:
        records = load_jsonl(path)
        if not records:
            continue
        first = records[0]
        rows.append({"file": Path(path).name, "model": first["model"], "condition": first["condition"], **score(records)})
        if args.breakdown:
            print(f"\n== {Path(path).name}")
            print_table(breakdown(records, "expression_id"), ["expression_id", "n", "literal", "idiomatic", "accuracy"])
            print()
            print_table(breakdown(records, "sentence_sufficient"), ["sentence_sufficient", "n", "literal", "idiomatic", "accuracy"])

    columns = ["model", "condition", "n", "accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci", "unparsed_or_refused"]
    print()
    print_table(rows, columns)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file"] + columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    main()
