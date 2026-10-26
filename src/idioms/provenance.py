"""Run fingerprints and checks shared by inference and evaluation."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .data import LABELS, read_jsonl

PROTOCOL_VERSION = 1


def fingerprint(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pipeline_fingerprint():
    root = Path(__file__).parent
    files = ("data.py", "models.py", "prompts.py", "run.py", "provenance.py")
    return fingerprint({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files})


def runtime_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("mlx-lm", "mlx", "transformers", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def make_manifest(model, examples, split, template, condition):
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "data_hash": fingerprint(examples), "pipeline_hash": pipeline_fingerprint(),
        "runtime": runtime_versions(), "model": model.name, "settings": model.settings,
        "split": split, "prompt": template.name, "prompt_hash": template.hash,
        "prompt_text": template.text, "condition": condition,
        "examples": [{"id": ex["id"], "expression_id": ex["expression_id"],
                      "gold": ex["label"], "sentence_sufficient": ex["sentence_sufficient"],
                      "input_hash": fingerprint(ex),
                      "input_prompt_hash": fingerprint(template.fill(ex, condition))} for ex in examples],
    }
    config = json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))
    return {"run_id": fingerprint(config), "config": config}


def manifest_path(path):
    return Path(path).with_suffix(".manifest.json")


def read_manifest(path):
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if config.get("protocol_version") != PROTOCOL_VERSION or manifest.get("run_id") != fingerprint(config):
        raise ValueError(f"{path}: invalid or unsupported run manifest")
    return manifest


def index_records(records):
    indexed = {}
    for row in records:
        example_id = row.get("id")
        if not isinstance(example_id, str) or not example_id or example_id in indexed:
            raise ValueError(f"missing or duplicate prediction ID: {example_id!r}")
        if row.get("gold") not in LABELS or "prediction" not in row or row["prediction"] not in (*LABELS, None):
            raise ValueError(f"{example_id}: invalid gold or prediction")
        indexed[example_id] = row
    if not indexed:
        raise ValueError("prediction set is empty")
    return indexed


def validate_predictions(records, manifest, complete=False):
    if not records:
        if complete:
            raise ValueError("prediction set is empty")
        return
    indexed = index_records(records)
    config = manifest["config"]
    expected = {ex["id"]: ex for ex in config["examples"]}
    if set(indexed) - expected.keys():
        raise ValueError("prediction file contains IDs outside this run")
    if complete and indexed.keys() != expected.keys():
        raise ValueError(f"incomplete run: {len(indexed)}/{len(expected)} predictions; resume before scoring")
    for example_id, row in indexed.items():
        if row.get("run_id") != manifest["run_id"]:
            raise ValueError(f"{example_id}: prediction belongs to a different run")
        for field in ("model", "settings", "split", "prompt", "prompt_hash", "condition", "data_hash", "pipeline_hash"):
            if row.get(field) != config[field]:
                raise ValueError(f"{example_id}: inconsistent {field}")
        for field in ("gold", "expression_id", "sentence_sufficient", "input_hash"):
            if row.get(field) != expected[example_id][field]:
                raise ValueError(f"{example_id}: inconsistent {field}")
        if fingerprint(row.get("input_prompt")) != expected[example_id]["input_prompt_hash"]:
            raise ValueError(f"{example_id}: inconsistent input prompt")
        status = row.get("status")
        if status not in ("ok", "unparsed", "refused") or (status == "ok") != (row["prediction"] in LABELS):
            raise ValueError(f"{example_id}: inconsistent prediction status")


def load_predictions(path, allow_legacy=False):
    records = read_jsonl(path)
    if manifest_path(path).exists():
        validate_predictions(records, read_manifest(path), complete=True)
    elif allow_legacy:
        index_records(records)
        for row in records:
            for field in ("model", "settings", "split", "prompt", "prompt_hash", "condition"):
                if row.get(field) != records[0].get(field):
                    raise ValueError(f"{path}: mixed legacy prediction metadata")
    else:
        raise ValueError(f"{path}: no run manifest; use --allow-legacy only to inspect historical trials")
    return records


def align_records(records_a, records_b):
    """Require a complete paired comparison with identical references."""
    a, b = index_records(records_a), index_records(records_b)
    if a.keys() != b.keys():
        raise ValueError("runs contain different example IDs; complete both runs before comparing")
    for example_id in a:
        for field in ("gold", "expression_id", "sentence_sufficient", "data_hash", "pipeline_hash"):
            if a[example_id].get(field) != b[example_id].get(field):
                raise ValueError(f"{example_id}: runs disagree on {field}")
    return a, b
