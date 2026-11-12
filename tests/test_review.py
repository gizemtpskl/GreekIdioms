import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from idioms import agreement, data
from idioms.agreement import agreement_summary
from idioms.data import check, read_jsonl, write_jsonl
from idioms.metrics import cohen_kappa
from idioms.models import AlwaysIdiomatic
from idioms.prompts import Template
from idioms.run import run
from idioms.review import (DECISION_FIELDS, LEGACY_DECISION_FIELDS, REVIEW_FIELDS, adjudicate, export_review,
                           import_review, read_csv, write_csv)

ROOT = Path(__file__).resolve().parent.parent


class TestReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data = self.folder / "source.jsonl"
        # Review tests need primary annotations; the canonical pilot is already reviewed.
        self.examples = read_jsonl(ROOT / "data/review/pilot/source.jsonl")[:2]
        for ex in self.examples:
            ex["annotator"] = "first"
            ex["notes"] = "Private annotation reasoning"
        write_jsonl(self.data, self.examples)
        self.export = self.folder / "export"
        export_review(self.data, self.export)
        self.manifest = self.export / "manifest.json"
        self.reviewed = self.folder / "reviewed.jsonl"
        self.decisions = self.folder / "decisions.csv"

    def fill_review(self, disagree=False):
        rows = read_csv(self.export / "reviewer.csv", REVIEW_FIELDS)
        by_id = {ex["id"]: ex for ex in self.examples}
        for row in rows:
            ex = by_id[row["id"]]
            row["label_reviewer2"] = ex["label"]
            row["sentence_sufficient_reviewer2"] = str(ex["sentence_sufficient"]).lower()
        if disagree:
            rows[0]["label_reviewer2"] = "ambiguous"
            rows[0]["notes_reviewer2"] = "Both readings are plausible."
        completed = self.folder / "completed.csv"
        write_csv(completed, REVIEW_FIELDS, rows)
        return completed

    def merge(self, completed, reviewer="second"):
        return import_review(self.data, completed, self.manifest, reviewer,
                             self.reviewed, self.decisions)

    def select(self, rows):
        path = self.folder / "selection.jsonl"
        write_jsonl(path, rows)
        with patch.object(data, "SPLITS", {"dev": path, "test": ROOT / "data/final/final.jsonl"}):
            return data.load_split("dev")

    def test_export_is_blind_and_preserves_greek(self):
        rows = read_csv(self.export / "reviewer.csv", REVIEW_FIELDS)
        self.assertEqual(set(rows[0]), set(REVIEW_FIELDS))
        self.assertNotIn("Private annotation reasoning", (self.export / "reviewer.csv").read_text())
        for row in rows:
            self.assertEqual(row["label_reviewer2"], "")
            self.assertEqual(row["sentence_sufficient_reviewer2"], "")
        self.assertEqual({r["target_sentence"] for r in rows},
                         {r["target_sentence"] for r in self.examples})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export_review(self.data, self.export)

    def test_agreement_is_unchanged_after_adjudication(self):
        completed = self.fill_review(disagree=True)
        self.assertEqual(self.merge(completed), (2, 1))
        before = read_jsonl(self.reviewed)
        self.assertEqual(check(before), [])
        with self.assertRaisesRegex(SystemExit, "pending"):
            self.select(before)
        summary_before = agreement_summary(before)
        rows = read_csv(self.decisions, DECISION_FIELDS)
        pending = next(ex for ex in before if ex["review_status"] == "pending")
        for field in ("expression", "target_sentence", "span", "context_before", "context_after", "notes_reviewer2"):
            self.assertEqual(rows[0][field], pending[field])
        self.assertEqual(rows[0]["disagreement_type"], "label")
        rows[0].update(final_label="ambiguous", final_sentence_sufficient="false",
                       adjudicator="first + second", adjudication_notes="Unresolved dual reading.")
        completed_decisions = self.folder / "completed_decisions.csv"
        write_csv(completed_decisions, DECISION_FIELDS, rows)
        final = self.folder / "final.jsonl"
        self.assertEqual(adjudicate(self.reviewed, completed_decisions, final), 1)
        after = read_jsonl(final)
        self.assertEqual(check(after), [])
        self.assertEqual(len(self.select(after)), 1)  # unresolved ambiguity excluded
        self.assertEqual(agreement_summary(after)["kappa"], summary_before["kappa"])
        self.assertEqual(agreement_summary(after)["agree"], summary_before["agree"])
        self.assertEqual(read_jsonl(self.data), self.examples)  # source remains untouched
        for ex in after:
            self.assertIn(ex["label_reviewer1"], ("literal", "idiomatic"))

    def test_import_rejects_changed_source(self):
        completed = self.fill_review()
        changed = [dict(ex) for ex in self.examples]
        changed[0]["label"] = "ambiguous"
        write_jsonl(self.data, changed)
        with self.assertRaisesRegex(ValueError, "changed since export"):
            self.merge(completed)
        self.assertFalse(self.reviewed.exists())

    def test_import_rejects_unfinished_and_same_reviewer(self):
        with self.assertRaisesRegex(ValueError, "label must"):
            self.merge(self.export / "reviewer.csv")
        completed = self.fill_review()
        with self.assertRaisesRegex(ValueError, "different person"):
            self.merge(completed, reviewer="first")
        self.assertFalse(self.reviewed.exists())

    def test_import_rejects_changed_passage_or_subset(self):
        completed = self.fill_review()
        original = read_csv(completed, REVIEW_FIELDS)
        for name, rows in [("changed", [dict(r) for r in original]),
                           ("missing", original[:1]), ("duplicate", original + original[:1])]:
            if name == "changed":
                rows[0]["context_before"] += " altered"
            path = self.folder / f"{name}.csv"
            write_csv(path, REVIEW_FIELDS, rows)
            with self.assertRaises(ValueError):
                self.merge(path)
        self.assertFalse(self.reviewed.exists())

    def test_sufficiency_disagreement_also_requires_resolution(self):
        completed = self.fill_review()
        rows = read_csv(completed, REVIEW_FIELDS)
        rows[0]["sentence_sufficient_reviewer2"] = (
            "false" if rows[0]["sentence_sufficient_reviewer2"] == "true" else "true")
        path = self.folder / "sufficiency.csv"
        write_csv(path, REVIEW_FIELDS, rows)
        self.assertEqual(self.merge(path), (2, 1))

    def test_adjudication_rejects_missing_decisions_and_changed_originals(self):
        self.merge(self.fill_review(disagree=True))
        original = read_csv(self.decisions, DECISION_FIELDS)
        for name in ("missing_label", "missing_sufficiency", "changed_original"):
            rows = [dict(r) for r in original]
            rows[0].update(final_label="literal", final_sentence_sufficient="true", adjudicator="first")
            if name == "missing_label":
                rows[0]["final_label"] = ""
            if name == "missing_sufficiency":
                rows[0]["final_sentence_sufficient"] = ""
            if name == "changed_original":
                rows[0].update(label_reviewer1="ambiguous", adjudication_notes="Reason")
            path = self.folder / f"{name}.csv"
            write_csv(path, DECISION_FIELDS, rows)
            with self.assertRaises(ValueError):
                adjudicate(self.reviewed, path, self.folder / "final.jsonl")
        self.assertFalse((self.folder / "final.jsonl").exists())

    def test_adjudication_allows_blank_attribution_and_notes(self):
        self.merge(self.fill_review(disagree=True))
        before = read_jsonl(self.reviewed)
        rows = read_csv(self.decisions, DECISION_FIELDS)
        rows[0].update(final_label="ambiguous", final_sentence_sufficient="false")
        self.assertEqual((rows[0]["adjudicator"], rows[0]["adjudication_notes"]), ("", ""))
        path = self.folder / "adjudication_completed.csv"
        write_csv(path, DECISION_FIELDS, rows)
        final = self.folder / "final.jsonl"
        self.assertEqual(adjudicate(self.reviewed, path, final), 1)
        after = read_jsonl(final)
        self.assertEqual(check(after), [])
        resolved = next(ex for ex in after if ex["review_status"] == "adjudicated")
        self.assertEqual((resolved["adjudicator"], resolved["adjudication_notes"]), ("", ""))
        self.assertEqual(agreement_summary(before)["kappa"], agreement_summary(after)["kappa"])
        self.assertEqual(len(self.select(after)), 1)
        without_metadata = dict(resolved)
        del without_metadata["adjudicator"], without_metadata["adjudication_notes"]
        self.assertEqual(check([without_metadata]), [])
        self.assertTrue(check([dict(resolved, adjudication_notes=42)]))

    def test_subset_import_leaves_other_examples_unreviewed(self):
        export_review(self.data, self.folder / "subset", ids=[self.examples[0]["id"]])
        self.export = self.folder / "subset"
        self.manifest = self.export / "manifest.json"
        self.assertEqual(self.merge(self.fill_review()), (1, 0))
        rows = read_jsonl(self.reviewed)
        self.assertNotIn("label_reviewer2", rows[1])
        self.assertEqual(agreement_summary(rows)["n"], 1)
        self.assertEqual(len(self.select(rows)), 2)

    def test_adjudication_rejects_modified_reference_columns(self):
        self.merge(self.fill_review(disagree=True))
        original = read_csv(self.decisions, DECISION_FIELDS)
        for field in ("target_sentence", "context_before", "span", "notes_reviewer2", "disagreement_type"):
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[0].update(final_label="ambiguous", final_sentence_sufficient="false",
                               adjudicator="first + second", adjudication_notes="Unresolved dual reading.")
                rows[0][field] += " changed"
                path = self.folder / f"changed_{field}.csv"
                write_csv(path, DECISION_FIELDS, rows)
                with self.assertRaisesRegex(ValueError, "read-only"):
                    adjudicate(self.reviewed, path, self.folder / "final.jsonl")
        self.assertFalse((self.folder / "final.jsonl").exists())

    def test_adjudication_accepts_existing_short_sheets(self):
        self.merge(self.fill_review(disagree=True))
        full_rows = read_csv(self.decisions, DECISION_FIELDS)
        rows = [{field: row[field] for field in LEGACY_DECISION_FIELDS} for row in full_rows]
        rows[0].update(final_label="ambiguous", final_sentence_sufficient="false",
                       adjudicator="first + second", adjudication_notes="Unresolved dual reading.")
        path = self.folder / "legacy_completed.csv"
        write_csv(path, LEGACY_DECISION_FIELDS, rows)
        final = self.folder / "final.jsonl"
        self.assertEqual(adjudicate(self.reviewed, path, final), 1)
        self.assertEqual(check(read_jsonl(final)), [])

    def test_agreement_requires_original_labels_and_reports_undefined_kappa(self):
        with self.assertRaisesRegex(ValueError, "original label_reviewer1"):
            agreement_summary([dict(self.examples[0], label_reviewer2="literal")])
        self.assertTrue(math.isnan(cohen_kappa(["literal"], ["literal"])))

    def test_review_metadata_cannot_hide_unresolved_disagreements(self):
        self.merge(self.fill_review(disagree=True))
        original = read_jsonl(self.reviewed)
        pending = next(ex for ex in original if ex["review_status"] == "pending")
        changes = [
            {"review_status": "agreed"},
            {"review_status": "unknown"},
            {"label_reviewer1": "invalid"},
            {"sentence_sufficient_reviewer1": "true"},
            {"sentence_sufficient_reviewer2": "false"},
            {"reviewer2": "first"},
            {"reviewer2": " "},
            {"notes_reviewer2": ""},
            {"label": "ambiguous"},
        ]
        for change in changes:
            with self.subTest(change=change):
                altered = dict(pending, **change)
                self.assertTrue(check([altered]))
                with self.assertRaises(ValueError):
                    run(AlwaysIdiomatic(), [altered], "dev", Template("basic_en"), "context",
                        progress=False, out_dir=self.folder / "predictions")
        incomplete = dict(pending)
        del incomplete["label_reviewer1"]
        self.assertTrue(check([incomplete]))
        with self.assertRaisesRegex(ValueError, "resolved binary"):
            run(AlwaysIdiomatic(), original, "dev", Template("basic_en"), "context",
                progress=False, out_dir=self.folder / "predictions")
        self.assertFalse((self.folder / "predictions").exists())

    def test_agreed_reviews_work_with_current_runner_and_resumption(self):
        self.assertEqual(self.merge(self.fill_review()), (2, 0))
        selected = self.select(read_jsonl(self.reviewed))
        self.assertEqual(check(selected), [])
        args = (AlwaysIdiomatic(), selected, "dev", Template("basic_en"), "context")
        path, count = run(*args, progress=False, out_dir=self.folder / "predictions")
        self.assertEqual(count, 2)
        before = path.read_bytes()
        self.assertEqual(run(*args, progress=False, out_dir=path.parent)[1], 0)
        self.assertEqual(path.read_bytes(), before)

    def test_agreement_cli_accepts_explicit_files_and_default_splits(self):
        self.merge(self.fill_review())
        for args in ([str(self.reviewed)], []):
            output = io.StringIO()
            with patch.object(agreement, "SPLITS", {"dev": self.reviewed}), \
                    contextlib.redirect_stdout(output):
                agreement.main(args)
            self.assertIn("double-annotated: 2/2", output.getvalue())
            self.assertIn("raw agreement:    1.000", output.getvalue())
        other = self.folder / "other.jsonl"
        rows = copy.deepcopy(read_jsonl(self.reviewed))
        for row in rows:
            row["id"] = "other-" + row["id"]
        write_jsonl(other, rows)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            agreement.main([str(self.reviewed), str(other)])
        self.assertIn("double-annotated: 4/4", output.getvalue())

    def test_reviewer_identity_is_required_and_trimmed(self):
        completed = self.fill_review()
        with self.assertRaisesRegex(ValueError, "different person"):
            self.merge(completed, reviewer=" first ")
        self.merge(completed, reviewer=" second ")
        self.assertEqual(read_jsonl(self.reviewed)[0]["reviewer2"], "second")
        missing = copy.deepcopy(self.examples)
        del missing[0]["annotator"]
        path = self.folder / "missing_annotator.jsonl"
        write_jsonl(path, missing)
        with self.assertRaisesRegex(ValueError, "primary annotator"):
            export_review(path, self.folder / "missing_export")


if __name__ == "__main__":
    unittest.main()
