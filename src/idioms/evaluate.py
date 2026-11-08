"""Scores and analyses for the saved predictions of a split.

    python -m idioms.evaluate --split test                  # one row per run
    python -m idioms.evaluate --split test --prompt use_en --details
    python -m idioms.evaluate --split dev --grid            # macro-F1, models x prompts

--details adds, for each model: the effect of adding context (McNemar test),
scores for sentence-sufficient vs. context-dependent examples, minimal-pair
accuracy and accuracy per expression.
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

from .data import load_split, minimal_pairs, read_jsonl
from .metrics import accuracy, bootstrap_ci, macro_f1, mcnemar_exact, paired_bootstrap_diff
from .models import BASELINE
from .run import PREDICTIONS
from .provenance import align_records, fingerprint, load_predictions, manifest_path


def load_runs(split, prompt=None, pred_dir=None, allow_legacy=False):
    """{(model, prompt, condition): records} for every prediction file of the split."""
    runs = {}
    legacy = []
    for path in sorted(Path(pred_dir or PREDICTIONS).glob(f"{split}__*.jsonl")):
        records = read_jsonl(path)
        if records and (prompt is None or records[0]["prompt"] == prompt or records[0]["model"] == BASELINE):
            records = load_predictions(path, allow_legacy=allow_legacy)
            if not manifest_path(path).exists():
                legacy.append(path)
            key = (records[0]["model"], records[0]["prompt"], records[0]["condition"])
            if key in runs:
                raise ValueError(f"duplicate run for {key}")
            if records[0].get("split") != split:
                raise ValueError(f"{path}: split metadata does not match the requested split")
            if runs:
                align_records(next(iter(runs.values())), records)
            runs[key] = records
    if legacy:
        warnings.warn(f"Inspecting {len(legacy)} historical runs without manifests; "
                      "their inputs, settings and parser version cannot be verified.", stacklevel=2)
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
    a, b = align_records(sentence_records, context_records)
    for example_id in a:
        for field in ("model", "settings", "prompt", "prompt_hash"):
            if a[example_id].get(field) != b[example_id].get(field):
                raise ValueError(f"{example_id}: context comparison differs in {field}")
    for records, expected in ((sentence_records, "sentence"), (context_records, "context")):
        if any("condition" in r and r["condition"] != expected for r in records):
            raise ValueError(f"expected the {expected} condition")
    ids = sorted(a)
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
    if not rows:
        print("No rows to display.")
        return
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in columns))


SUMMARY_COLUMNS = ["model", "prompt", "condition", "n", "accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci",
                   "predicted_literal", "unreadable"]


def summary_rows(runs):
    return [{"model": m, "prompt": p, "condition": c, **scores(records)} for (m, p, c), records in runs.items()]


def grid_rows(runs):
    """One row per model, one column per prompt: macro-F1 as 'sentence / context'.

    The baseline ignores the prompt, so its score is shown under every prompt.
    """
    f1 = {key: macro_f1(*gold_and_pred(records)) for key, records in runs.items()}
    prompts = sorted({p for _, p, _ in runs})
    models = [BASELINE] * any(m == BASELINE for m, _, _ in runs) + sorted({m for m, _, _ in runs if m != BASELINE})

    def cell(model, prompt):
        found = {c: v for (m, p, c), v in f1.items() if m == model and (p == prompt or model == BASELINE)}
        return " / ".join(f"{found[c]:.2f}" if c in found else "-" for c in ("sentence", "context"))

    return prompts, [{"model": m, **{p: cell(m, p) for p in prompts}} for m in models]


def print_grid(runs):
    prompts, rows = grid_rows(runs)
    print("macro-F1 (sentence / context)")
    print_table(rows, ["model", *prompts])


def validate_reference(runs, examples):
    """Ensure current dataset metadata can be attached to the saved runs."""
    reference = [{"id": ex["id"], "gold": ex["label"], "prediction": None,
                  "expression_id": ex["expression_id"], "sentence_sufficient": ex["sentence_sufficient"]}
                 for ex in examples]
    for records in runs.values():
        align_records(reference, [{k: v for k, v in r.items() if k not in ("data_hash", "pipeline_hash")}
                                  for r in records])
        if records[0].get("data_hash") and records[0]["data_hash"] != fingerprint(examples):
            raise ValueError("dataset changed since inference; cannot attach current examples to these predictions")


def print_details(runs, examples):
    validate_reference(runs, examples)
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
    ap.add_argument("--grid", action="store_true", help="compact view: macro-F1 per model (rows) and prompt (columns)")
    ap.add_argument("--results-dir", type=Path, default=PREDICTIONS.parent)
    ap.add_argument("--allow-legacy", action="store_true", help="inspect historical runs without manifests")
    args = ap.parse_args(argv)

    try:
        runs = load_runs(args.split, args.prompt, args.results_dir / "predictions", args.allow_legacy)
        if not runs:
            raise SystemExit(f"no predictions for {args.split} in {args.results_dir / 'predictions'}")
        validate_reference(runs, load_split(args.split))
        if args.grid:
            print_grid(runs)
        else:
            print_table(summary_rows(runs), SUMMARY_COLUMNS)
        if args.details:
            print_details(runs, load_split(args.split))
    except (ValueError, OSError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
