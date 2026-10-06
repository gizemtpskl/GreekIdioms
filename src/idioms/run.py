"""Run one model under one condition on one split and save predictions.

    python -m idioms.run --data data/sample/toy.jsonl --splits data/sample/splits.json \
        --split test --model anthropic:claude-opus-5-5 --condition context

Predictions go to <split>__<model>__<prompt name>__<condition>.jsonl. They are
appended one line at a time, and examples already in the output file are
skipped, so an interrupted run can simply be restarted. If the prompt file has
been edited since, the run stops instead of mixing two prompt versions.
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import load_jsonl, load_splits, select, validate, write_jsonl
from .models import OK, get_backend, parse_label
from .prompts import CONDITIONS, build_prompt, load_template, template_hash


def output_path(out_dir, split, model, prompt_file, condition):
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", model)
    return Path(out_dir) / f"{split or 'all'}__{slug}__{Path(prompt_file).stem}__{condition}.jsonl"


def already_done(path, prompt_hash):
    if not path.exists():
        return set()
    records = load_jsonl(path)
    if any(rec["prompt_hash"] != prompt_hash for rec in records):
        raise SystemExit(
            f"{path} was produced with a different version of the prompt file.\n"
            "Delete it to rerun, or save the edited prompt under a new name."
        )
    return {rec["id"] for rec in records}


def load_examples(data_path, splits_path=None, split=None, limit=None):
    """Validated, binary-labelled examples of one split."""
    examples = load_jsonl(data_path)
    problems = validate(examples)
    if problems:
        unlabelled = sum(not ex.get("label") for ex in examples)
        hint = (f"\n{unlabelled} of {len(examples)} examples have no label yet: the file is still being annotated."
                if unlabelled else "")
        shown = "\n".join(problems[:10]) + (f"\n... and {len(problems) - 10} more" if len(problems) > 10 else "")
        raise SystemExit(f"{data_path} has {len(problems)} problems, so nothing was run.{hint}\n"
                         f"Full list: python -m idioms.validate {data_path}\n\n{shown}")
    splits = load_splits(splits_path) if splits_path else None
    examples = select(examples, splits, split)
    return examples[:limit] if limit else examples


def run_predictions(backend, model, examples, prompt_file, condition, out, verbose=True):
    """Classify every example not yet in `out` and append the predictions to it."""
    template = load_template(prompt_file)
    done = already_done(out, template_hash(template))
    todo = [ex for ex in examples if ex["id"] not in done]
    if verbose:
        print(f"{len(examples)} examples, {len(done)} already done, running {len(todo)} -> {out}")

    for i, ex in enumerate(todo, 1):
        prompt = build_prompt(template, ex, condition)
        start = time.time()
        raw, status, *extra = backend.classify(prompt)
        prediction = parse_label(raw) if status == OK else None
        record = {
            "id": ex["id"],
            "expression_id": ex["expression_id"],
            "gold": ex["label"],
            "prediction": prediction,
            "status": status if prediction or status != OK else "unparsed",
            "raw_output": raw,
            "sentence_sufficient": ex.get("sentence_sufficient"),
            "model": model,
            "condition": condition,
            "settings": backend.settings(),
            "prompt_file": str(prompt_file),
            "prompt_hash": template_hash(template),
            "seconds": round(time.time() - start, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra[0] if extra else {}),
        }
        write_jsonl(out, [record], mode="a")
        if verbose:
            print(f"[{i}/{len(todo)}] {ex['id']}: gold={ex['label']} pred={prediction} ({record['status']})")
    return len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="annotated examples (JSONL)")
    ap.add_argument("--splits", help="splits.json mapping split name -> expression ids")
    ap.add_argument("--split", help="which split to run on (dev/test); omit to use all examples")
    ap.add_argument("--model", required=True, help="backend spec, e.g. anthropic:claude-opus-5-5")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--prompt", default="prompts/classify.txt")
    ap.add_argument("--out-dir", default="results/predictions")
    ap.add_argument("--effort", default="low", help="effort for Claude models; 'none' to omit the parameter")
    ap.add_argument("--limit", type=int, help="only run the first N examples (for quick checks)")
    args = ap.parse_args(argv)

    examples = load_examples(args.data, args.splits, args.split, args.limit)
    effort = None if args.effort == "none" else args.effort
    backend = get_backend(args.model, effort=effort)
    out = output_path(args.out_dir, args.split, args.model, args.prompt, args.condition)
    run_predictions(backend, args.model, examples, args.prompt, args.condition, out)


if __name__ == "__main__":
    main()
