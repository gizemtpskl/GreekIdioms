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
from pathlib import Path
from statistics import mean

from .data import ROOT, load_split
from .evaluate import SUMMARY_COLUMNS, fmt, print_details, print_table, summary_rows
from .models import BASELINE, DEFAULT_METHOD, METHODS, MODELS, free_memory, load_model
from .prompts import CONDITIONS, Template, prompt_names
from .run import run
from .provenance import PROTOCOL_VERSION, fingerprint, load_predictions, pipeline_fingerprint, runtime_versions

RESULTS = ROOT / "results"


def run_all(split, model_keys, method, templates, results_dir=RESULTS, expected_settings=None):
    examples = load_split(split)
    print(f"{split}: {len(examples)} examples; models {', '.join(model_keys)}; "
          f"prompts {', '.join(t.name for t in templates)}; method {method}\n")
    runs = {}
    for key in [BASELINE, *model_keys]:
        model = load_model(key, method)
        if expected_settings is not None and key != BASELINE and model.settings != expected_settings.get(key):
            raise ValueError(f"{key}: model settings differ from the saved development selection")
        for template in templates[:1] if key == BASELINE else templates:  # the baseline ignores the prompt
            for condition in CONDITIONS:
                path, n_new = run(model, examples, split, template, condition, progress=False,
                                  out_dir=Path(results_dir) / "predictions")
                records = load_predictions(path)
                runs[(model.name, template.name, condition)] = records
                print(f"  {model.name:14} {template.name:12} {condition:9} {n_new} new predictions")
        del model
        free_memory()
    return runs


def write_summary(rows, split, results_dir=RESULTS):
    path = Path(results_dir) / f"{split}_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows([{c: fmt(r[c]) for c in SUMMARY_COLUMNS} for r in rows])
    print(f"\nsaved to {path}")


def dev(model_keys, method, prompt_list, results_dir=RESULTS):
    if not model_keys or len(model_keys) != len(set(model_keys)):
        raise ValueError("choose a nonempty list of distinct models")
    if not prompt_list or len(prompt_list) != len(set(prompt_list)):
        raise ValueError("choose a nonempty list of distinct prompts")
    templates = [Template(name) for name in prompt_list]  # fails early if a prompt is malformed
    runs = run_all("dev", model_keys, method, templates, results_dir)
    rows = summary_rows(runs)
    print()
    print_table(rows, SUMMARY_COLUMNS)
    write_summary(rows, "dev", results_dir)

    ranking = sorted(((mean(r["macro_f1"] for r in rows if r["prompt"] == t.name and r["model"] != BASELINE), t)
                      for t in templates), key=lambda x: (-x[0], x[1].name))
    print("\nPrompts ranked by macro-F1 (mean over models and conditions):")
    for score, t in ranking:
        print(f"  {t.name:12} {score:.3f}")
    best_score, best = ranking[0]
    if len(ranking) > 1 and ranking[1][0] == best_score:
        print("  note: tie at the top; the alphabetically first prompt was taken (mention this in the report)")
    settings = {key: runs[(key if method == "generate" else f"{key}-score", best.name, "sentence")][0]["settings"]
                for key in model_keys}
    chosen_path = Path(results_dir) / "chosen_prompt.json"
    chosen_path.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION,
                                      "pipeline_hash": pipeline_fingerprint(),
                                      "runtime": runtime_versions(),
                                      "dev_data_hash": fingerprint(load_split("dev")),
                                      "prompt": best.name, "prompt_hash": best.hash, "method": method,
                                      "models": model_keys, "model_settings": settings,
                                      "dev_macro_f1": round(best_score, 4)}, indent=2) + "\n")
    print(f"\nChosen prompt: {best.name}  (saved to {chosen_path})")


def test(model_keys=None, results_dir=RESULTS):
    chosen_path = Path(results_dir) / "chosen_prompt.json"
    if not chosen_path.exists():
        raise SystemExit("No prompt chosen yet. Run `python -m idioms.experiment dev` first.")
    chosen = json.loads(chosen_path.read_text())
    if chosen.get("protocol_version") != PROTOCOL_VERSION or chosen.get("pipeline_hash") != pipeline_fingerprint():
        raise SystemExit("The saved selection predates this inference pipeline. Run dev in a new results directory first.")
    if chosen.get("dev_data_hash") != fingerprint(load_split("dev")):
        raise SystemExit("Development data changed since prompt selection. Run dev in a new results directory first.")
    if chosen.get("runtime") != runtime_versions():
        raise SystemExit("Runtime versions changed since prompt selection. Run dev in a new results directory first.")
    saved_models = chosen.get("models", [])
    if not saved_models or len(set(saved_models)) != len(saved_models) or any(k not in MODELS for k in saved_models):
        raise SystemExit("Invalid saved model selection; rerun dev.")
    if model_keys is not None and (len(model_keys) != len(set(model_keys)) or set(model_keys) != set(saved_models)):
        raise SystemExit(f"Test must use the models selected on dev: {', '.join(saved_models)}")
    if chosen.get("method") not in METHODS or set(chosen.get("model_settings", {})) != set(saved_models):
        raise SystemExit("Invalid saved method or settings; rerun dev.")
    template = Template(chosen["prompt"])
    if template.hash != chosen["prompt_hash"]:
        raise SystemExit(f"prompts/{template.name}.txt changed after it was chosen on dev. "
                         "Restore it, or rerun the dev step.")
    print(f"Using the prompt chosen on dev: {template.name} (method {chosen['method']})\n")
    runs = run_all("test", saved_models, chosen["method"], [template], results_dir,
                   expected_settings=chosen["model_settings"])
    rows = summary_rows(runs)
    print()
    print_table(rows, SUMMARY_COLUMNS)
    write_summary(rows, "test", results_dir)
    print_details(runs, load_split("test"))
    print(f"\nFigures: python -m idioms.figures --split test --prompt {template.name} --results-dir {results_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["dev", "test"])
    ap.add_argument("--models", nargs="+", choices=list(MODELS), help="test defaults to the models saved by dev")
    ap.add_argument("--method", choices=METHODS, help=f"dev step only (default: {DEFAULT_METHOD}); test reuses dev's")
    ap.add_argument("--prompts", nargs="+", help="dev step only (default: all prompts)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    if args.step == "test" and (args.method is not None or args.prompts is not None):
        ap.error("test reuses the saved method and prompt; --method and --prompts are dev-only")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.step == "dev":
            dev(args.models or list(MODELS), args.method or DEFAULT_METHOD, args.prompts or prompt_names(), args.results_dir)
        else:
            test(args.models, args.results_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
