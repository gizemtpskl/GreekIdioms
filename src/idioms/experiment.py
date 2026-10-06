"""Run the whole experiment on one split and summarise it.

    # 1. dev: every model x every prompt x both conditions (+ baseline), then pick the prompt
    python -m idioms.experiment --data data/pilot/examples.jsonl --splits data/pilot/splits.json --split dev

    # 2. test: only the prompt chosen on dev
    python -m idioms.experiment --data data/pilot/examples.jsonl --splits data/pilot/splits.json \
        --split test --prompts prompts/use_en.txt

The prompt is chosen by the rule in the README: the highest macro-F1 averaged
over both models and both conditions on dev. On the test split only one prompt
may be run, so the choice can't be influenced by test results.

Each model is loaded once; runs already finished are skipped, so the command
can be restarted after an interruption. The summary table is also written to
results/<split>_summary.csv.
"""

import argparse
import csv
import gc
from glob import glob
from pathlib import Path
from statistics import mean

from .evaluate import compare, fmt, print_table, score
from .data import load_jsonl
from .models import answer_words, get_backend
from .prompts import CONDITIONS, load_template
from .run import load_examples, output_path, run_predictions

DEFAULT_MODELS = [
    "mlx:ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx",
    "mlx:mlx-community/Llama-3.1-8B-Instruct-4bit",
]
BASELINE = "baseline:always_idiomatic"


def free_memory():
    """Release the previous model before loading the next one (16 GB machines need this)."""
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def short_name(model):
    return model.split("/")[-1] if "/" in model else model


def choose_prompt(rows, prompts):
    """Rank prompts by macro-F1 averaged over all non-baseline models and both conditions."""
    ranking = []
    for prompt in prompts:
        name = Path(prompt).stem
        scores = [r["macro_f1"] for r in rows if r["prompt"] == name and r["model"] != BASELINE]
        ranking.append((mean(scores), name, prompt))
    ranking.sort(key=lambda t: (-t[0], t[1]))
    return ranking


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True, help="dev or test")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompts", nargs="+", default=sorted(glob("prompts/*.txt")))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--out-dir", default="results/predictions")
    ap.add_argument("--summary-dir", default="results")
    ap.add_argument("--limit", type=int, help="only the first N examples (for quick checks)")
    args = ap.parse_args(argv)

    if args.split == "test" and len(args.prompts) > 1:
        raise SystemExit("On the test split, run only the prompt chosen on dev: --prompts prompts/<chosen>.txt")
    for prompt in args.prompts:
        answer_words(load_template(prompt))  # fail now, not after an hour of running, if a prompt is malformed

    examples = load_examples(args.data, args.splits, args.split, args.limit)
    if not examples:
        raise SystemExit(f"no binary-labelled examples in split {args.split!r}")
    print(f"{len(examples)} examples in {args.split}; {len(args.models)} models x {len(args.prompts)} prompts x {len(args.conditions)} conditions\n")

    runs = []  # (model, prompt, condition, path)
    for model in [BASELINE] + args.models:
        backend = get_backend(model)
        prompts = args.prompts[:1] if model == BASELINE else args.prompts  # the baseline ignores the prompt
        for prompt in prompts:
            for condition in args.conditions:
                path = output_path(args.out_dir, args.split, model, prompt, condition)
                n_new = run_predictions(backend, model, examples, prompt, condition, path, verbose=False)
                print(f"  {short_name(model):40} {Path(prompt).stem:12} {condition:9} {n_new} new predictions")
                runs.append((model, prompt, condition, path))
        del backend
        free_memory()

    rows = []
    for model, prompt, condition, path in runs:
        records = load_jsonl(path)
        row = {"model": model, "prompt": Path(prompt).stem, "condition": condition, **score(records)}
        rows.append(row)
    print()
    print_table([{**r, "model": short_name(r["model"])} for r in rows],
                ["model", "prompt", "condition", "n", "accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci", "unparsed_or_refused"])

    summary = Path(args.summary_dir) / f"{args.split}_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten to {summary}")

    best_prompt = args.prompts[0]
    if len(args.prompts) > 1:
        ranking = choose_prompt(rows, args.prompts)
        print("\nPrompt ranking (macro-F1 averaged over models and conditions):")
        for avg, name, _ in ranking:
            print(f"  {name:12} {fmt(avg)}")
        best_prompt = ranking[0][2]
        if len(ranking) > 1 and ranking[0][0] == ranking[1][0]:
            print("  note: the top prompts are tied; the alphabetically first was taken. Mention this in the report.")
        print(f"\nChosen prompt: {best_prompt}\nNext, run on test:\n"
              f"  python -m idioms.experiment --data {args.data} --splits {args.splits} --split test --prompts {best_prompt}")

    if set(CONDITIONS) <= set(args.conditions):
        for model in args.models:
            print(f"\n=== Context effect: {short_name(model)}, prompt {Path(best_prompt).stem} (A = sentence only, B = with context)")
            compare(*(output_path(args.out_dir, args.split, model, best_prompt, c) for c in CONDITIONS))


if __name__ == "__main__":
    main()
