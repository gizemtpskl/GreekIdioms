"""Scores and analyses for the saved predictions of a split.

    python -m idioms.evaluate --split test                  # one row per run
    python -m idioms.evaluate --split test --prompt use_en --details

--details adds, for each model: the effect of adding context (McNemar test),
scores for sentence-sufficient vs. context-dependent examples, minimal-pair
accuracy and accuracy per expression.
"""

import argparse
from collections import defaultdict

from .data import load_split, minimal_pairs, read_jsonl
from .metrics import accuracy, bootstrap_ci, macro_f1, mcnemar_exact, paired_bootstrap_diff
from .models import BASELINE
from .run import PREDICTIONS


def load_runs(split, prompt=None):
    """{(model, prompt, condition): records} for every prediction file of the split."""
    runs = {}
    for path in sorted(PREDICTIONS.glob(f"{split}__*.jsonl")):
        records = read_jsonl(path)
        if records and (prompt is None or records[0]["prompt"] == prompt or records[0]["model"] == BASELINE):
            runs[(records[0]["model"], records[0]["prompt"], records[0]["condition"])] = records
    return runs


def gold_and_pred(records):
    return [r["gold"] for r in records], [r["prediction"] for r in records]


def scores(records):
    gold, pred = gold_and_pred(records)
    acc_lo, acc_hi = bootstrap_ci(gold, pred, accuracy)
    f1_lo, f1_hi = bootstrap_ci(gold, pred, macro_f1)
    return {
        "n": len(records),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci": (acc_lo, acc_hi),
        "macro_f1": macro_f1(gold, pred),
        "macro_f1_ci": (f1_lo, f1_hi),
        "unreadable": sum(p is None for p in pred),
        "predicted_literal": sum(p == "literal" for p in pred),
    }


def by_group(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {value: accuracy(*gold_and_pred(group)) for value, group in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def pair_accuracy(records, pairs):
    """Share of minimal pairs where both examples are classified correctly."""
    correct = {r["id"]: r["prediction"] == r["gold"] for r in records}
    usable = [(a, b) for a, b in pairs if a in correct and b in correct]
    return sum(correct[a] and correct[b] for a, b in usable) / len(usable) if usable else float("nan")


def context_effect(sentence_records, context_records):
    """How adding context changed the predictions on the same examples."""
    a = {r["id"]: r for r in sentence_records}
    b = {r["id"]: r for r in context_records}
    ids = sorted(a.keys() & b.keys())
    gold = [a[i]["gold"] for i in ids]
    pred_a = [a[i]["prediction"] for i in ids]
    pred_b = [b[i]["prediction"] for i in ids]
    right_a = [g == p for g, p in zip(gold, pred_a)]
    right_b = [g == p for g, p in zip(gold, pred_b)]
    fixed = [i for i, x, y in zip(ids, right_a, right_b) if not x and y]
    broken = [i for i, x, y in zip(ids, right_a, right_b) if x and not y]
    return {
        "n": len(ids),
        "accuracy_gain": accuracy(gold, pred_b) - accuracy(gold, pred_a),
        "accuracy_gain_ci": paired_bootstrap_diff(gold, pred_a, pred_b, accuracy),
        "macro_f1_gain": macro_f1(gold, pred_b) - macro_f1(gold, pred_a),
        "fixed": fixed,
        "broken": broken,
        "mcnemar_p": mcnemar_exact(len(broken), len(fixed)),
    }


def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, tuple):
        return f"[{value[0]:.2f}, {value[1]:.2f}]"
    return str(value)


def print_table(rows, columns):
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in columns))


SUMMARY_COLUMNS = ["model", "prompt", "condition", "n", "accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci",
                   "predicted_literal", "unreadable"]


def summary_rows(runs):
    return [{"model": m, "prompt": p, "condition": c, **scores(records)} for (m, p, c), records in runs.items()]


def print_details(runs, examples):
    pairs = minimal_pairs(examples)
    names = {ex["expression_id"]: ex["expression"] for ex in examples}
    models = sorted({(m, p) for m, p, _ in runs if m != BASELINE})
    for model, prompt in models:
        sentence, context = runs.get((model, prompt, "sentence")), runs.get((model, prompt, "context"))
        print(f"\n=== {model}, prompt {prompt}")
        if sentence and context:
            effect = context_effect(sentence, context)
            print(f"Adding context: accuracy {effect['accuracy_gain']:+.3f} (95% CI {fmt(effect['accuracy_gain_ci'])}), "
                  f"macro-F1 {effect['macro_f1_gain']:+.3f}; fixed {len(effect['fixed'])}, broke {len(effect['broken'])}, "
                  f"McNemar p = {effect['mcnemar_p']:.4f}")
        rows = []
        for condition, records in (("sentence", sentence), ("context", context)):
            if records:
                acc = by_group(records, "sentence_sufficient")
                rows.append({"condition": condition, "sentence_sufficient": acc.get(True, float("nan")),
                             "needs_context": acc.get(False, float("nan")),
                             "minimal_pairs_both_right": pair_accuracy(records, pairs)})
        print()
        print_table(rows, ["condition", "sentence_sufficient", "needs_context", "minimal_pairs_both_right"])
        per_expression = {c: by_group(r, "expression_id") for c, r in (("sentence", sentence), ("context", context)) if r}
        print()
        print_table([{"expression": f"{e} {names.get(e, '')}", **{c: per_expression[c][e] for c in per_expression}}
                     for e in sorted(names)], ["expression", *per_expression])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--prompt", help="only runs with this prompt (the baseline is always shown)")
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args(argv)

    runs = load_runs(args.split, args.prompt)
    if not runs:
        raise SystemExit(f"no predictions for {args.split} in {PREDICTIONS}")
    print_table(summary_rows(runs), SUMMARY_COLUMNS)
    if args.details:
        print_details(runs, load_split(args.split))


if __name__ == "__main__":
    main()
