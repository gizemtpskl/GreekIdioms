"""The whole experiment in two commands.

    python -m idioms.experiment dev     # 1. choose the prompt on the pilot data
    python -m idioms.experiment test    # 2. evaluate on the final data with that prompt

dev  runs the baseline and both models with every prompt in prompts/, in both
     conditions, on data/pilot/pilot.jsonl. The prompt with the highest macro-F1,
     averaged over both models and both conditions, is saved to
     results/chosen_prompt.json together with a fingerprint of its text.

test runs the baseline and both models with the chosen prompt only, on
     data/final/final.jsonl, and prints the full analysis. It refuses to run if
     no prompt has been chosen or the chosen prompt was edited afterwards.

Finished predictions are kept, so either command can be restarted after an
interruption. Each model needs about 5 GB of memory: close other heavy apps.
"""

import argparse
import csv
import json
from statistics import mean

from .data import ROOT, load_split
from .evaluate import SUMMARY_COLUMNS, fmt, load_runs, print_details, print_table, summary_rows
from .models import BASELINE, METHODS, MODELS, free_memory, load_model
from .prompts import CONDITIONS, Template, prompt_names
from .run import run

RESULTS = ROOT / "results"
CHOSEN = RESULTS / "chosen_prompt.json"


def run_all(split, model_keys, method, templates):
    examples = load_split(split)
    print(f"{split}: {len(examples)} examples; models {', '.join(model_keys)}; "
          f"prompts {', '.join(t.name for t in templates)}; method {method}\n")
    names = []
    for key in [BASELINE, *model_keys]:
        model = load_model(key, method)
        names.append(model.name)
        for template in templates[:1] if key == BASELINE else templates:  # the baseline ignores the prompt
            for condition in CONDITIONS:
                _, n_new = run(model, examples, split, template, condition, progress=False)
                print(f"  {model.name:14} {template.name:12} {condition:9} {n_new} new predictions")
        del model
        free_memory()
    prompts = {t.name for t in templates}
    return {k: v for k, v in load_runs(split).items() if k[0] in names and (k[0] == BASELINE or k[1] in prompts)}


def write_summary(rows, split):
    path = RESULTS / f"{split}_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows([{c: fmt(r[c]) for c in SUMMARY_COLUMNS} for r in rows])
    print(f"\nsaved to {path.relative_to(ROOT)}")


def dev(model_keys, method, prompt_list):
    templates = [Template(name) for name in prompt_list]  # fails early if a prompt is malformed
    runs = run_all("dev", model_keys, method, templates)
    rows = summary_rows(runs)
    print()
    print_table(rows, SUMMARY_COLUMNS)
    write_summary(rows, "dev")

    ranking = sorted(((mean(r["macro_f1"] for r in rows if r["prompt"] == t.name and r["model"] != BASELINE), t)
                      for t in templates), key=lambda x: (-x[0], x[1].name))
    print("\nPrompts ranked by macro-F1 (mean over models and conditions):")
    for score, t in ranking:
        print(f"  {t.name:12} {score:.3f}")
    best_score, best = ranking[0]
    if len(ranking) > 1 and ranking[1][0] == best_score:
        print("  note: tie at the top; the alphabetically first prompt was taken (mention this in the report)")
    CHOSEN.write_text(json.dumps({"prompt": best.name, "prompt_hash": best.hash, "method": method,
                                  "models": model_keys, "dev_macro_f1": round(best_score, 4)}, indent=2) + "\n")
    print(f"\nChosen prompt: {best.name}  (saved to {CHOSEN.relative_to(ROOT)})\nNext: python -m idioms.experiment test")


def test(model_keys):
    if not CHOSEN.exists():
        raise SystemExit("No prompt chosen yet. Run `python -m idioms.experiment dev` first.")
    chosen = json.loads(CHOSEN.read_text())
    template = Template(chosen["prompt"])
    if template.hash != chosen["prompt_hash"]:
        raise SystemExit(f"prompts/{template.name}.txt changed after it was chosen on dev. "
                         "Restore it, or rerun the dev step.")
    print(f"Using the prompt chosen on dev: {template.name} (method {chosen['method']})\n")
    runs = run_all("test", model_keys, chosen["method"], [template])
    rows = summary_rows(runs)
    print()
    print_table(rows, SUMMARY_COLUMNS)
    write_summary(rows, "test")
    print_details(runs, load_split("test"))
    print(f"\nFigures: python -m idioms.figures --split test --prompt {template.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["dev", "test"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--method", default="generate", choices=METHODS, help="dev step only; test reuses dev's")
    ap.add_argument("--prompts", nargs="+", default=prompt_names(), help="dev step only")
    args = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    if args.step == "dev":
        dev(args.models, args.method, args.prompts)
    else:
        test(args.models)


if __name__ == "__main__":
    main()
