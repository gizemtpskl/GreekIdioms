"""Figures for the report, made from the prediction files of one split and prompt.

    python -m idioms.figures --split test --prompt use_en --data data/pilot/examples.jsonl

Writes to results/figures/ (PNG for slides, PDF for the report):
  scores_<split>.*           accuracy and macro-F1 per model and condition, 95% bootstrap CIs,
                             with the always-idiomatic baseline as a dashed line
  per_expression_<split>.*   accuracy per expression, sentence only vs. with context
  context_effect_<split>.*   errors fixed and introduced by adding context, with McNemar p
  per_expression_<split>.csv the numbers behind the per-expression figure
"""

import argparse
import csv
from collections import defaultdict
from glob import glob
from pathlib import Path

from .data import load_jsonl
from .experiment import BASELINE, DEFAULT_MODELS, short_name
from .metrics import accuracy, bootstrap_ci, macro_f1, mcnemar_exact

# Colours checked with the dataviz palette validator (colour-blind safe, light background).
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CONDITION_STYLE = {
    "sentence": {"color": "#2a78d6", "hatch": "", "marker": "o", "label": "Sentence only"},
    "context": {"color": "#eb6834", "hatch": "///", "marker": "D", "label": "With context"},
}
FIXED, BROKEN = "#4a3aa7", "#eda100"
DISPLAY_NAMES = {
    "mlx:ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx": "Krikri 8B",
    "mlx:mlx-community/Llama-3.1-8B-Instruct-4bit": "Llama 3.1 8B",
    "mlx-score:ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx": "Krikri 8B (scored)",
    "mlx-score:mlx-community/Llama-3.1-8B-Instruct-4bit": "Llama 3.1 8B (scored)",
}


def display_name(model):
    return DISPLAY_NAMES.get(model, short_name(model))


def load_runs(pred_dir, split, prompt):
    """{model: {condition: records}} for one split and prompt, plus the baseline records."""
    runs = defaultdict(dict)
    for path in glob(f"{pred_dir}/{split}__*__{prompt}__*.jsonl"):
        records = load_jsonl(path)
        if records and records[0]["model"] != BASELINE:
            runs[records[0]["model"]][records[0]["condition"]] = records
    if not runs:
        raise SystemExit(f"no prediction files for split {split!r} and prompt {prompt!r} in {pred_dir}")
    baseline_files = sorted(glob(f"{pred_dir}/{split}__baseline-always-idiomatic__*__sentence.jsonl"))
    baseline = load_jsonl(baseline_files[0]) if baseline_files else None
    order = [m for m in DEFAULT_MODELS if m in runs] + sorted(m for m in runs if m not in DEFAULT_MODELS)
    return {m: runs[m] for m in order}, baseline


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelcolor=INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_dir) / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)


def scores_figure(plt, runs, baseline, split):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, facecolor=SURFACE)
    width = 0.36
    for ax, (metric_name, metric) in zip(axes, [("Accuracy", accuracy), ("Macro-F1", macro_f1)]):
        style_axes(ax)
        for i, (model, by_condition) in enumerate(runs.items()):
            for j, (condition, style) in enumerate(CONDITION_STYLE.items()):
                records = by_condition.get(condition)
                if not records:
                    continue
                gold = [r["gold"] for r in records]
                pred = [r["prediction"] for r in records]
                value = metric(gold, pred)
                lo, hi = bootstrap_ci(gold, pred, metric)
                x = i + (j - 0.5) * (width + 0.02)  # small gap between the two bars
                ax.bar(x, value, width, color=style["color"], hatch=style["hatch"], edgecolor=SURFACE,
                       linewidth=0, label=style["label"] if i == 0 else None)
                ax.errorbar(x, value, yerr=[[value - lo], [hi - value]], fmt="none", ecolor=INK_SOFT,
                            elinewidth=1, capsize=3)
                ax.text(x, hi + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        if baseline:
            base = metric([r["gold"] for r in baseline], [r["prediction"] for r in baseline])
            ax.axhline(base, color=INK_SOFT, linestyle="--", linewidth=1, label="Always-idiomatic baseline")
            ax.text(-0.58, base + 0.01, f"{base:.2f}", ha="left", va="bottom", fontsize=8, color=INK_SOFT)
        ax.set_xticks(range(len(runs)), [display_name(m) for m in runs])
        ax.set_xlim(-0.6, len(runs) - 0.4)
        ax.set_ylim(0, 1.12)
        ax.set_title(metric_name, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Score (95% bootstrap CI)", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Scores on the {split} split", y=1.1, color=INK, fontsize=12)
    return fig


def per_expression_rows(runs, expressions):
    rows = []
    for model, by_condition in runs.items():
        groups = defaultdict(lambda: defaultdict(list))
        for condition, records in by_condition.items():
            for r in records:
                groups[r["expression_id"]][condition].append(r)
        for expr_id in sorted(groups):
            row = {"model": display_name(model), "expression_id": expr_id,
                   "expression": expressions.get(expr_id, expr_id)}
            for condition in CONDITION_STYLE:
                recs = groups[expr_id].get(condition, [])
                row["n"] = len(recs) or row.get("n", 0)
                row[f"accuracy_{condition}"] = accuracy([r["gold"] for r in recs], [r["prediction"] for r in recs]) if recs else None
            rows.append(row)
    return rows


def per_expression_figure(plt, rows, split):
    models = list(dict.fromkeys(r["model"] for r in rows))
    expr_ids = list(dict.fromkeys(r["expression_id"] for r in rows))
    labels = {r["expression_id"]: f"{r['expression']} (n={r['n']})" for r in rows}
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + 1.5, 0.45 * len(expr_ids) + 1.6),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ax, model in zip(axes[0], models):
        style_axes(ax, grid_axis="x")
        for y, expr_id in enumerate(expr_ids):
            row = next(r for r in rows if r["model"] == model and r["expression_id"] == expr_id)
            values = [row[f"accuracy_{c}"] for c in CONDITION_STYLE]
            if None not in values:
                ax.plot(values, [y - 0.12, y + 0.12], color=GRID, linewidth=2, zorder=1)
            for offset, (condition, style) in zip((-0.12, 0.12), CONDITION_STYLE.items()):
                value = row[f"accuracy_{condition}"]
                if value is not None:  # small vertical offset so equal values don't hide each other
                    ax.scatter(value, y + offset, s=55, marker=style["marker"], color=style["color"],
                               edgecolor=SURFACE, linewidth=1.5, zorder=2,
                               label=style["label"] if y == 0 else None)
        ax.set_yticks(range(len(expr_ids)), [labels[e] for e in expr_ids])
        ax.set_ylim(len(expr_ids) - 0.5, -0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("Accuracy", color=INK)
        ax.set_title(model, color=INK, fontsize=11, loc="left")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Accuracy per expression ({split})", color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def context_effect_figure(plt, runs, split):
    stats = []
    for model, by_condition in runs.items():
        if not {"sentence", "context"} <= by_condition.keys():
            continue
        sentence = {r["id"]: r for r in by_condition["sentence"]}
        context = {r["id"]: r for r in by_condition["context"]}
        ids = sentence.keys() & context.keys()
        correct = lambda r: r["prediction"] == r["gold"]
        fixed = sum(not correct(sentence[i]) and correct(context[i]) for i in ids)
        broken = sum(correct(sentence[i]) and not correct(context[i]) for i in ids)
        stats.append((display_name(model), fixed, broken, mcnemar_exact(broken, fixed)))
    if not stats:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 0.8 * len(stats) + 1.6), facecolor=SURFACE)
    style_axes(ax, grid_axis="x")
    limit = max(max(f, b) for _, f, b, _ in stats) + 1
    for y, (name, fixed, broken, p) in enumerate(stats):
        ax.barh(y, fixed, height=0.5, color=FIXED, label="Errors fixed by context" if y == 0 else None)
        ax.barh(y, -broken, height=0.5, color=BROKEN, label="Errors introduced by context" if y == 0 else None)
        ax.text(fixed + limit * 0.02, y, str(fixed), va="center", ha="left", fontsize=9, color=INK)
        ax.text(-broken - limit * 0.02, y, str(broken), va="center", ha="right", fontsize=9, color=INK)
        ax.text(limit * 1.15, y, f"McNemar p = {p:.3f}", va="center", ha="left", fontsize=9, color=INK_SOFT)
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.set_yticks(range(len(stats)), [s[0] for s in stats])
    ax.set_ylim(len(stats) - 0.5, -0.5)
    ax.set_xlim(-limit, limit)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [str(abs(int(t))) for t in ticks])
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Number of examples", color=INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"What adding context changed ({split})", y=1.12, color=INK, fontsize=12)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--prompt", required=True, help="prompt name, e.g. use_en")
    ap.add_argument("--data", help="examples file, to show expressions instead of their ids")
    ap.add_argument("--pred-dir", default="results/predictions")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "hatch.color": SURFACE, "hatch.linewidth": 1.2})
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    runs, baseline = load_runs(args.pred_dir, args.split, args.prompt)
    expressions = {ex["expression_id"]: ex["expression"] for ex in load_jsonl(args.data)} if args.data else {}

    save(scores_figure(plt, runs, baseline, args.split), args.out_dir, f"scores_{args.split}")
    rows = per_expression_rows(runs, expressions)
    save(per_expression_figure(plt, rows, args.split), args.out_dir, f"per_expression_{args.split}")
    with open(Path(args.out_dir) / f"per_expression_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = context_effect_figure(plt, runs, args.split)
    if fig:
        save(fig, args.out_dir, f"context_effect_{args.split}")
    print(f"figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
