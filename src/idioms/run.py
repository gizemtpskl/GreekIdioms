"""Run one model, prompt and condition, with validated resumption.

    python -m idioms.run --split dev --model baseline --prompt basic_en --condition sentence \
        --out-dir results/verified/predictions

Companion manifests fingerprint inputs, settings, prompt, inference code and
runtime. Historical files without manifests are preserved and cannot be resumed.
Choose a new output directory for new runs.
"""

import argparse
import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .data import ROOT, LABELS, check, load_split, read_jsonl, write_jsonl
from .models import BASELINE, METHODS, MODELS, load_model
from .prompts import CONDITIONS, Template, prompt_names
from .provenance import (fingerprint, make_manifest, manifest_path, read_manifest,
                         validate_predictions)

PREDICTIONS = ROOT / "results/predictions"


def prediction_path(split, model_name, prompt_name, condition, out_dir=None):
    return Path(out_dir or PREDICTIONS) / f"{split}__{model_name}__{prompt_name}__{condition}.jsonl"


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another process is writing {path}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare_run(path, manifest):
    meta = manifest_path(path)
    if meta.exists():
        previous = read_manifest(path)
        if previous != manifest:
            changed = [k for k in manifest["config"] if previous["config"].get(k) != manifest["config"][k]]
            raise ValueError(f"{path.name}: incompatible run ({', '.join(changed)}); use a new output directory")
    elif path.exists():
        raise ValueError(f"{path.name}: historical predictions have no manifest; use a new output directory")
    else:
        with meta.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        raise ValueError(f"{path.name}: incomplete last line; inspect the file before resuming")
    done = read_jsonl(path) if path.exists() else []
    validate_predictions(done, manifest)
    return {row["id"] for row in done}


def run(model, examples, split, template, condition, progress=True, out_dir=None):
    """Return (prediction path, new prediction count). Failed calls remain unfinished."""
    problems = check(examples)
    if not examples or problems or any(ex["label"] not in LABELS or ex.get("review_status") == "pending" for ex in examples):
        raise ValueError("run requires nonempty, valid, resolved binary examples: " + "; ".join(problems[:5]))
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    path = prediction_path(split, model.name, template.name, condition, out_dir)
    manifest = make_manifest(model, examples, split, template, condition)
    with run_lock(path):
        done_ids = prepare_run(path, manifest)
        todo = [ex for ex in examples if ex["id"] not in done_ids]
        for i, ex in enumerate(todo, 1):
            start = time.monotonic()
            prompt = template.fill(ex, condition)
            try:
                output = model.predict(prompt, template.answer_words)
                if not isinstance(output, dict) or output.get("prediction") not in (*LABELS, None) or "prediction" not in output:
                    raise ValueError("backend returned an invalid prediction")
                if not isinstance(output.get("raw_output"), str):
                    raise ValueError("backend returned invalid raw output")
                status = output.get("status", "ok" if output["prediction"] is not None else "unparsed")
                if status not in ("ok", "unparsed", "refused") or (status == "ok") != (output["prediction"] in LABELS):
                    raise ValueError("backend returned an inconsistent status")
            except Exception as exc:
                # Local inference failures are not classification mistakes.
                # Do not blindly retry memory or configuration failures.
                error_path = path.parent / "failures" / path.name
                write_jsonl(error_path, [{"id": ex["id"], "run_id": manifest["run_id"],
                                        "status": "request_error", "error_type": type(exc).__name__,
                                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}], mode="a")
                raise RuntimeError(f"{ex['id']}: {type(exc).__name__}; failure logged to {error_path}. "
                                   "Fix the cause and rerun the same command; completed predictions are retained.") from exc
            record = {
                "id": ex["id"], "expression_id": ex["expression_id"], "gold": ex["label"],
                "prediction": output["prediction"], "raw_output": output["raw_output"], "status": status,
                "sentence_sufficient": ex["sentence_sufficient"],
                "split": split, "model": model.name, "settings": manifest["config"]["settings"],
                "prompt": template.name, "prompt_hash": template.hash, "condition": condition,
                "run_id": manifest["run_id"], "data_hash": manifest["config"]["data_hash"],
                "pipeline_hash": manifest["config"]["pipeline_hash"],
                "input_hash": fingerprint(ex), "input_prompt": prompt,
                "seconds": round(time.monotonic() - start, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if "logprobs" in output:
                record["logprobs"] = output["logprobs"]
            write_jsonl(path, [record], mode="a")
            if progress:
                print(f"[{i}/{len(todo)}] {ex['id']}: gold={ex['label']} prediction={output['prediction']} ({status})")
    return path, len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--model", required=True, choices=[BASELINE, *MODELS])
    ap.add_argument("--method", default="generate", choices=METHODS)
    ap.add_argument("--prompt", required=True, choices=prompt_names())
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--out-dir", type=Path, default=PREDICTIONS)
    args = ap.parse_args(argv)
    try:
        examples = load_split(args.split)
        template = Template(args.prompt)
        model = load_model(args.model, args.method)
        path, n_new = run(model, examples, args.split, template, args.condition, out_dir=args.out_dir)
        print(f"{n_new} new predictions -> {path}")
    except (ValueError, OSError, RuntimeError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
