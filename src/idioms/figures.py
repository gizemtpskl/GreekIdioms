"""Figures for the report from the saved predictions of one split and prompt.

    python -m idioms.figures --split test --prompt use_en

Writes PNG (slides) and PDF (report) files to results/figures/:
  scores_<split>          accuracy and macro-F1 per model and condition, 95% bootstrap CIs,
                          always-idiomatic baseline as a dashed line
  example_types_<split>   accuracy on sentence-sufficient vs. context-dependent examples,
                          and the share of minimal pairs with both halves right
  context_effect_<split>  errors fixed and introduced by adding context, with McNemar p
  per_expression_<split>  accuracy per expression with and without context
                          (numbers also saved as per_expression_<split>.csv)
"""

import argparse
import csv

from .data import ROOT, load_split, minimal_pairs
from .evaluate import by_group, context_effect, gold_and_pred, load_runs, pair_accuracy
from .metrics import accuracy, bootstrap_ci, macro_f1
from .models import BASELINE, MODELS

OUT = ROOT / "results/figures"
# colours checked with a colour-blind palette validator; light background, print-friendly
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "sentence": {"color": "#2a78d6", "hatch": "", "marker": "o", "label": "Sentence only"},
    "context": {"color": "#eb6834", "hatch": "///", "marker": "D", "label": "With context"},
}
FIXED, BROKEN = "#4a3aa7", "#eda100"
DISPLAY = {"krikri": "Krikri 8B", "llama": "Llama 3.1 8B", "krikri-score": "Krikri 8B (scored)",
           "llama-score": "Llama 3.1 8B (scored)"}


def model_order(runs):
    names = {m for m, _, _ in runs if m != BASELINE}
    preferred = [n for key in MODELS for n in (key, f"{key}-score") if n in names]
    return preferred + sorted(names - set(preferred))


def style_axes(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelcolor=INK)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend_and_title(fig, ax, title, ncol=3):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.07, color=INK, fontsize=12)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)


def grouped_bars(ax, models, values_for, value_ci_for=None):
    """Two bars (sentence / context) per model; values_for(model, condition) -> value or None."""
    width = 0.36
    for i, model in enumerate(models):
        for j, (condition, style) in enumerate(STYLE.items()):
            value = values_for(model, condition)
            if value is None:
                continue
            x = i + (j - 0.5) * (width + 0.02)
            ax.bar(x, value, width, color=style["color"], hatch=style["hatch"], linewidth=0,
                   label=style["label"] if i == 0 else None)
            top = value
            if value_ci_for:
                lo, hi = value_ci_for(model, condition)
                ax.errorbar(x, value, yerr=[[value - lo], [hi - value]], fmt="none", ecolor=INK_SOFT, elinewidth=1, capsize=3)
                top = hi
            ax.text(x, top + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(models)), [DISPLAY.get(m, m) for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1.12)


def scores_figure(plt, runs, prompt, split):
    models = model_order(runs)
    baseline = next((r for (m, _, c), r in runs.items() if m == BASELINE and c == "sentence"), None)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (title, metric) in zip(axes, [("Accuracy", accuracy), ("Macro-F1", macro_f1)]):
        style_axes(ax)
        get = lambda m, c: runs.get((m, prompt, c))
        grouped_bars(ax, models,
                     lambda m, c: metric(*gold_and_pred(get(m, c))) if get(m, c) else None,
                     lambda m, c: bootstrap_ci(*gold_and_pred(get(m, c)), metric))
        if baseline:
            base = metric(*gold_and_pred(baseline))
            ax.axhline(base, color=INK_SOFT, linestyle="--", linewidth=1, label="Always-idiomatic baseline")
            ax.text(-0.58, base + 0.01, f"{base:.2f}", ha="left", va="bottom", fontsize=8, color=INK_SOFT)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Score (95% bootstrap CI)", color=INK)
    legend_and_title(fig, axes[0], f"Scores on the {split} split (prompt: {prompt})")
    return fig


def example_types_figure(plt, runs, prompt, split, pairs):
    models = model_order(runs)
    panels = [
        ("Sentence alone is enough", lambda r: by_group(r, "sentence_sufficient").get(True)),
        ("Needs context", lambda r: by_group(r, "sentence_sufficient").get(False)),
        ("Minimal pairs: both right", lambda r: pair_accuracy(r, pairs)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (title, measure) in zip(axes, panels):
        style_axes(ax)
        grouped_bars(ax, models, lambda m, c: measure(runs[(m, prompt, c)]) if (m, prompt, c) in runs else None)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Accuracy / share of pairs", color=INK)
    legend_and_title(fig, axes[0], f"Accuracy by example type ({split})", ncol=2)
    return fig


def context_effect_figure(plt, runs, prompt, split):
    stats = []
    for model in model_order(runs):
        if (model, prompt, "sentence") in runs and (model, prompt, "context") in runs:
            e = context_effect(runs[(model, prompt, "sentence")], runs[(model, prompt, "context")])
            stats.append((DISPLAY.get(model, model), len(e["fixed"]), len(e["broken"]), e["mcnemar_p"]))
    if not stats:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 0.8 * len(stats) + 1.6), facecolor=SURFACE)
    style_axes(ax, grid="x")
    limit = max(max(f, b) for _, f, b, _ in stats) + 1
    for y, (name, fixed, broken, p) in enumerate(stats):
        ax.barh(y, fixed, height=0.5, color=FIXED, label="Errors fixed by context" if y == 0 else None)
        ax.barh(y, -broken, height=0.5, color=BROKEN, label="Errors introduced by context" if y == 0 else None)
        ax.text(fixed + limit * 0.02, y, str(fixed), va="center", ha="left", fontsize=9, color=INK)
        ax.text(-broken - limit * 0.02, y, str(broken), va="center", ha="right", fontsize=9, color=INK)
        ax.text(limit * 1.12, y, f"McNemar p = {p:.3f}", va="center", ha="left", fontsize=9, color=INK_SOFT)
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.set_yticks(range(len(stats)), [s[0] for s in stats])
    ax.set_ylim(len(stats) - 0.5, -0.5)
    ax.set_xlim(-limit, limit)
    ticks = [t for t in ax.get_xticks() if -limit <= t <= limit]
    ax.set_xticks(ticks, [str(abs(int(t))) for t in ticks])
    ax.set_xlabel("Number of examples", color=INK)
    legend_and_title(fig, ax, f"What adding context changed ({split})", ncol=2)
    return fig


def per_expression(runs, prompt, examples):
    names = {ex["expression_id"]: ex["expression"] for ex in examples}
    counts = {e: sum(ex["expression_id"] == e for ex in examples) for e in names}
    rows = []
    for model in model_order(runs):
        acc = {c: by_group(runs[(model, prompt, c)], "expression_id") for c in STYLE if (model, prompt, c) in runs}
        for e in sorted(names):
            rows.append({"model": DISPLAY.get(model, model), "expression_id": e, "expression": names[e],
                         "n": counts[e], **{f"accuracy_{c}": acc[c].get(e) for c in acc}})
    return rows


def per_expression_figure(plt, rows, split):
    models = list(dict.fromkeys(r["model"] for r in rows))
    exprs = list(dict.fromkeys(r["expression_id"] for r in rows))
    label = {r["expression_id"]: f"{r['expression']} (n={r['n']})" for r in rows}
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + 2, 0.32 * len(exprs) + 1.6),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ax, model in zip(axes[0], models):
        style_axes(ax, grid="x")
        for y, e in enumerate(exprs):
            row = next(r for r in rows if r["model"] == model and r["expression_id"] == e)
            values = [row.get(f"accuracy_{c}") for c in STYLE]
            if None not in values:
                ax.plot(values, [y - 0.12, y + 0.12], color=GRID, linewidth=2, zorder=1)
            for offset, (c, style) in zip((-0.12, 0.12), STYLE.items()):  # offset so equal values stay visible
                if row.get(f"accuracy_{c}") is not None:
                    ax.scatter(row[f"accuracy_{c}"], y + offset, s=40, marker=style["marker"], color=style["color"],
                               edgecolor=SURFACE, linewidth=1.2, zorder=2, label=style["label"] if y == 0 else None)
        ax.set_yticks(range(len(exprs)), [label[e] for e in exprs])
        ax.set_ylim(len(exprs) - 0.5, -0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("Accuracy", color=INK)
        ax.set_title(model, color=INK, fontsize=11, loc="left")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Accuracy per expression ({split})", color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "hatch.color": SURFACE, "hatch.linewidth": 1.2})
    runs = load_runs(args.split, args.prompt)
    if not any(m != BASELINE for m, _, _ in runs):
        raise SystemExit(f"no model predictions for {args.split} with prompt {args.prompt!r}")
    examples = load_split(args.split)
    OUT.mkdir(parents=True, exist_ok=True)

    save(scores_figure(plt, runs, args.prompt, args.split), f"scores_{args.split}")
    save(example_types_figure(plt, runs, args.prompt, args.split, minimal_pairs(examples)), f"example_types_{args.split}")
    fig = context_effect_figure(plt, runs, args.prompt, args.split)
    if fig:
        save(fig, f"context_effect_{args.split}")
    rows = per_expression(runs, args.prompt, examples)
    save(per_expression_figure(plt, rows, args.split), f"per_expression_{args.split}")
    with open(OUT / f"per_expression_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"figures written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
