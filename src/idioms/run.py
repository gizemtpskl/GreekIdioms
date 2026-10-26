"""Run one model with one prompt in one condition on one split, and save the predictions.

    python -m idioms.run --split dev --model krikri --prompt use_en --condition context
    python -m idioms.run --split dev --model llama --method score --prompt use_el --condition sentence

Predictions go to results/predictions/<split>__<model>__<prompt>__<condition>.jsonl,
one line per example. Examples already in that file are skipped, so a run that
stopped halfway can be restarted with the same command. If the prompt file has
changed since, the run stops instead of mixing two versions of the prompt.
"""

import argparse
import time
from datetime import datetime, timezone

from .data import ROOT, load_split, read_jsonl, write_jsonl
from .models import BASELINE, METHODS, MODELS, load_model
from .prompts import CONDITIONS, Template, prompt_names

PREDICTIONS = ROOT / "results/predictions"


def prediction_path(split, model_name, prompt_name, condition):
    return PREDICTIONS / f"{split}__{model_name}__{prompt_name}__{condition}.jsonl"


def run(model, examples, split, template, condition, progress=True):
    """Predict every example not yet in the output file; returns (path, number of new predictions)."""
    path = prediction_path(split, model.name, template.name, condition)
    done = read_jsonl(path) if path.exists() else []
    if any(r["prompt_hash"] != template.hash for r in done):
        raise SystemExit(f"{path.name} was made with an earlier version of prompts/{template.name}.txt.\n"
                         "Delete the file to rerun, or save the changed prompt under a new name.")
    done_ids = {r["id"] for r in done}
    todo = [ex for ex in examples if ex["id"] not in done_ids]
    for i, ex in enumerate(todo, 1):
        start = time.time()
        output = model.predict(template.fill(ex, condition), template.answer_words)
        record = {
            "id": ex["id"],
            "expression_id": ex["expression_id"],
            "gold": ex["label"],
            **output,
            "sentence_sufficient": ex["sentence_sufficient"],
            "split": split,
            "model": model.name,
            "settings": model.settings,
            "prompt": template.name,
            "prompt_hash": template.hash,
            "condition": condition,
            "seconds": round(time.time() - start, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_jsonl(path, [record], mode="a")
        if progress:
            print(f"[{i}/{len(todo)}] {ex['id']}: gold={ex['label']} prediction={output['prediction']}")
    return path, len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--model", required=True, choices=[BASELINE, *MODELS])
    ap.add_argument("--method", default="generate", choices=METHODS)
    ap.add_argument("--prompt", required=True, choices=prompt_names())
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    args = ap.parse_args(argv)

    examples = load_split(args.split)
    model = load_model(args.model, args.method)
    path, n_new = run(model, examples, args.split, Template(args.prompt), args.condition)
    print(f"{n_new} new predictions -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
