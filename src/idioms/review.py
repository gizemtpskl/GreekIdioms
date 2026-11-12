"""Export blind CSV reviews, import independent labels, and record adjudication.

Run `python -m idioms.review --help` for the three workflow commands.
All commands create new files and refuse to overwrite existing outputs.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from .data import ANNOTATION_LABELS, check, read_jsonl, write_jsonl
from .provenance import fingerprint

PASSAGE_FIELDS = ("id", "expression", "target_sentence", "span", "context_before", "context_after")
REVIEW_FIELDS = PASSAGE_FIELDS + ("label_reviewer2", "sentence_sufficient_reviewer2", "notes_reviewer2")
LEGACY_DECISION_FIELDS = (
    "id", "label_reviewer1", "label_reviewer2", "sentence_sufficient_reviewer1",
    "sentence_sufficient_reviewer2", "final_label", "final_sentence_sufficient",
    "adjudicator", "adjudication_notes",
)
FINAL_DECISION_FIELDS = LEGACY_DECISION_FIELDS[5:]
DECISION_REFERENCE_FIELDS = PASSAGE_FIELDS + LEGACY_DECISION_FIELDS[1:5] + (
    "notes_reviewer2", "disagreement_type",
)
DECISION_FIELDS = DECISION_REFERENCE_FIELDS + FINAL_DECISION_FIELDS


def checked_data(path):
    records = read_jsonl(path)
    problems = check(records)
    if problems:
        raise ValueError("invalid dataset:\n" + "\n".join(problems))
    return records


def require_new(*paths):
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be different")
    for path in paths:
        if Path(path).exists():
            raise ValueError(f"refusing to overwrite {path}; choose a new output path")


def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path, fields, alternatives=()):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in [list(fields), *(list(option) for option in alternatives)]:
            raise ValueError(f"{path}: unexpected columns; preserve the exported headers")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate example IDs")
    return rows


def parse_bool(value, where):
    value = value.strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"{where}: sentence sufficiency must be true or false")
    return value == "true"


def parse_annotation(value, where):
    value = value.strip().lower()
    if value not in ANNOTATION_LABELS:
        raise ValueError(f"{where}: label must be literal, idiomatic, or ambiguous")
    return value


def export_review(data, out_dir, ids=None, seed=0):
    records = checked_data(data)
    by_id = {ex["id"]: ex for ex in records}
    selected = list(by_id) if ids is None else list(ids)
    if not selected or len(selected) != len(set(selected)) or set(selected) - by_id.keys():
        raise ValueError("select a nonempty set of unique IDs present in the dataset")
    for example_id in selected:
        annotator = by_id[example_id].get("annotator")
        if not isinstance(annotator, str) or not annotator.strip():
            raise ValueError(f"{example_id}: a primary annotator identifier is required for review")
    if any(by_id[i].get("label_reviewer2") for i in selected):
        raise ValueError("selected examples already have second-reviewer labels")
    random.Random(seed).shuffle(selected)
    out_dir = Path(out_dir)
    csv_path, manifest_path = out_dir / "reviewer.csv", out_dir / "manifest.json"
    require_new(csv_path, manifest_path)
    rows = [dict.fromkeys(REVIEW_FIELDS, "") for _ in selected]
    for row, example_id in zip(rows, selected):
        row.update({field: by_id[example_id].get(field, "") for field in PASSAGE_FIELDS})
    write_csv(csv_path, REVIEW_FIELDS, rows)
    manifest = {"version": 1, "dataset_sha256": fingerprint(records), "ids": selected, "seed": seed}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return len(rows)


def decision_row(ex):
    row = dict.fromkeys(DECISION_FIELDS, "")
    for field in PASSAGE_FIELDS + ("label_reviewer1", "label_reviewer2", "notes_reviewer2"):
        row[field] = ex.get(field, "")
    for field in ("sentence_sufficient_reviewer1", "sentence_sufficient_reviewer2"):
        if ex.get(field) is not None:
            row[field] = str(ex[field]).lower()
    differences = []
    if ex["label_reviewer1"] != ex["label_reviewer2"]:
        differences.append("label")
    if ex.get("sentence_sufficient_reviewer1") != ex.get("sentence_sufficient_reviewer2"):
        differences.append("sentence_sufficiency")
    row["disagreement_type"] = "; ".join(differences)
    return row


def import_review(data, review_csv, manifest_path, reviewer, out, decisions_out):
    records = checked_data(data)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("dataset_sha256") != fingerprint(records):
        raise ValueError("dataset changed since export; reconcile the changes and export a new review")
    rows = read_csv(review_csv, REVIEW_FIELDS)
    if {r["id"] for r in rows} != set(manifest["ids"]):
        raise ValueError("review IDs must match the exported subset exactly")
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer identifier must not be empty")
    by_id = {ex["id"]: ex for ex in records}
    decisions = []
    for row in rows:
        ex = by_id[row["id"]]
        annotator = ex.get("annotator")
        if not isinstance(annotator, str) or not annotator.strip():
            raise ValueError(f"{ex['id']}: a primary annotator identifier is required for review")
        if reviewer == annotator.strip():
            raise ValueError(f"{ex['id']}: the second reviewer must be a different person")
        if ex.get("label_reviewer2"):
            raise ValueError(f"{ex['id']}: already reviewed")
        if any(row[f] != ex.get(f, "") for f in PASSAGE_FIELDS):
            raise ValueError(f"{ex['id']}: passage fields changed; record corrections in notes instead")
        label = parse_annotation(row["label_reviewer2"], ex["id"])
        sufficient = parse_bool(row["sentence_sufficient_reviewer2"], ex["id"])
        notes = row["notes_reviewer2"].strip()
        if label == "ambiguous" and not notes:
            raise ValueError(f"{ex['id']}: explain the ambiguous label in notes_reviewer2")
        ex.update(
            label_reviewer1=ex["label"],
            sentence_sufficient_reviewer1=ex.get("sentence_sufficient"),
            label_reviewer2=label,
            sentence_sufficient_reviewer2=sufficient,
            reviewer2=reviewer,
            notes_reviewer2=notes,
        )
        disagrees = label != ex["label"] or sufficient != ex.get("sentence_sufficient")
        ex["review_status"] = "pending" if disagrees else "agreed"
        if disagrees:
            decisions.append(decision_row(ex))
    require_new(out, decisions_out)
    write_jsonl(out, records, mode="x")
    write_csv(decisions_out, DECISION_FIELDS, decisions)
    return len(rows), len(decisions)


def adjudicate(data, decisions_csv, out):
    records = checked_data(data)
    pending = {ex["id"]: ex for ex in records if ex.get("review_status") == "pending"}
    rows = read_csv(decisions_csv, DECISION_FIELDS, alternatives=(LEGACY_DECISION_FIELDS,))
    if {r["id"] for r in rows} != pending.keys():
        raise ValueError("decision IDs must match all pending examples exactly")
    for row in rows:
        ex = pending[row["id"]]
        expected = decision_row(ex)
        for field in DECISION_REFERENCE_FIELDS:
            if field in row and row[field] != expected[field]:
                raise ValueError(f"{ex['id']}: read-only passage or annotation fields were changed: {field}")
        label = parse_annotation(row["final_label"], ex["id"])
        sufficient = parse_bool(row["final_sentence_sufficient"], ex["id"])
        ex.update(
            label=label, sentence_sufficient=sufficient, review_status="adjudicated",
            adjudicator=row["adjudicator"].strip(),
            adjudication_notes=row["adjudication_notes"].strip(),
        )
    require_new(out)
    write_jsonl(out, records, mode="x")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="create a blind CSV and a dataset fingerprint")
    export.add_argument("data")
    export.add_argument("--out-dir", required=True)
    export.add_argument("--ids", nargs="+", help="optional subset; otherwise export every example")
    export.add_argument("--seed", type=int, default=0)
    merge = commands.add_parser("import", help="preserve independent labels and list disagreements")
    merge.add_argument("data")
    merge.add_argument("review_csv")
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--reviewer", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--decisions-out", required=True)
    resolve = commands.add_parser("adjudicate", help="apply documented human decisions to a new file")
    resolve.add_argument("data")
    resolve.add_argument("decisions_csv")
    resolve.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "export":
            n = export_review(args.data, args.out_dir, args.ids, args.seed)
            print(f"exported {n} examples to {args.out_dir}/reviewer.csv")
        elif args.command == "import":
            n, pending = import_review(args.data, args.review_csv, args.manifest, args.reviewer,
                                       args.out, args.decisions_out)
            print(f"imported {n} independent reviews; {pending} examples require adjudication")
        else:
            n = adjudicate(args.data, args.decisions_csv, args.out)
            print(f"recorded {n} adjudications; independent labels preserved")
    except (ValueError, OSError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
