"""Regression tests for the review findings; no MLX weights or network required."""

import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from idioms import data, experiment
from idioms.evaluate import context_effect, load_runs, validate_reference
from idioms.models import AlwaysIdiomatic, MLXModel, parse_answer
from idioms.prompts import Template
from idioms.provenance import (PROTOCOL_VERSION, fingerprint, load_predictions,
                               manifest_path, pipeline_fingerprint, runtime_versions)
from idioms.run import prediction_path, run, run_lock


class FakeModel:
    name = "krikri"

    def __init__(self, fail_on=None, output=None):
        self.settings = {"method": "generate", "max_tokens": 10}
        self.calls = 0
        self.fail_on = fail_on
        self.output = output or {"prediction": "idiomatic", "raw_output": "idiomatic"}

    def predict(self, prompt, answer_words):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("simulated local inference failure")
        return dict(self.output)


class TestRunSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # These tests vary labels before review to exercise run identity checks.
        self.examples = data.read_jsonl(data.ROOT / "data/review/pilot/source.jsonl")[:2]
        self.template = Template("basic_en")

    def execute(self, model=None, examples=None, template=None, condition="sentence", directory=None):
        return run(model or FakeModel(), self.examples if examples is None else examples, "dev",
                   template or self.template, condition, progress=False, out_dir=directory or self.root)

    def test_resume_skips_only_verified_completed_predictions(self):
        model = FakeModel()
        path, count = self.execute(model)
        self.assertEqual((count, model.calls), (2, 2))
        self.assertTrue(manifest_path(path).exists())
        before = path.read_bytes()
        self.assertEqual(self.execute(model)[1], 0)
        self.assertEqual(model.calls, 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(load_predictions(path)), 2)

    def test_changed_data_settings_prompt_or_runtime_cannot_resume(self):
        path, _ = self.execute()
        before = path.read_bytes()
        for field, value in [("label", "literal"), ("context_before", "Changed context."),
                             ("target_sentence", self.examples[0]["target_sentence"] + " Changed.")]:
            changed = copy.deepcopy(self.examples)
            changed[0][field] = value
            model = FakeModel()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "incompatible run"):
                self.execute(model, examples=changed)
            self.assertEqual(model.calls, 0)
        model = FakeModel()
        model.settings["max_tokens"] = 20
        with self.assertRaisesRegex(ValueError, "settings"):
            self.execute(model)
        template = copy.copy(self.template)
        template.text += "\nChanged instructions."
        with self.assertRaisesRegex(ValueError, "prompt_text"):
            self.execute(template=template)
        with patch("idioms.provenance.runtime_versions", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "runtime"):
                self.execute()
        self.assertEqual(path.read_bytes(), before)

    def test_failure_leaves_item_unfinished_and_resume_completes_it(self):
        with self.assertRaisesRegex(RuntimeError, "failure logged"):
            self.execute(FakeModel(fail_on=2))
        path = prediction_path("dev", "krikri", "basic_en", "sentence", self.root)
        self.assertEqual(len(data.read_jsonl(path)), 1)
        failure = data.read_jsonl(self.root / "failures" / path.name)[0]
        self.assertEqual((failure["id"], failure["status"]), (self.examples[1]["id"], "request_error"))
        with self.assertRaisesRegex(ValueError, "incomplete run"):
            load_predictions(path)
        model = FakeModel()
        self.assertEqual(self.execute(model)[1], 1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(load_predictions(path)), 2)

    def test_unparsed_and_refused_are_finished_wrong_answers(self):
        for status in ("unparsed", "refused"):
            model = FakeModel(output={"prediction": None, "raw_output": "cannot answer", "status": status})
            path, _ = self.execute(model, directory=self.root / status)
            rows = load_predictions(path)
            self.assertTrue(all(r["prediction"] is None and r["status"] == status for r in rows))
            self.assertEqual(self.execute(model, directory=self.root / status)[1], 0)
            self.assertEqual(model.calls, 2)

    def test_invalid_backend_result_cannot_overwrite_record_metadata(self):
        output = {"prediction": "idiomatic", "raw_output": "idiomatic", "gold": "wrong", "id": "wrong"}
        path, _ = self.execute(FakeModel(output=output))
        rows = load_predictions(path)
        self.assertEqual(rows[0]["id"], self.examples[0]["id"])
        self.assertEqual(rows[0]["gold"], self.examples[0]["label"])
        with self.assertRaises(RuntimeError):
            self.execute(FakeModel(output={"prediction": "other", "raw_output": "other"}),
                         directory=self.root / "invalid")

    def test_legacy_predictions_are_preserved_and_not_resumed(self):
        path = prediction_path("dev", "krikri", "basic_en", "sentence", self.root)
        data.write_jsonl(path, [{"id": self.examples[0]["id"], "gold": "idiomatic", "prediction": "idiomatic"}])
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "historical predictions"):
            self.execute()
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "no run manifest"):
            load_predictions(path)
        self.assertEqual(len(load_predictions(path, allow_legacy=True)), 1)

    def test_duplicate_corrupt_and_incomplete_records_cannot_resume(self):
        path, _ = self.execute()
        original = data.read_jsonl(path)
        cases = [original + original[:1], [dict(original[0], gold="literal"), original[1]],
                 [dict(original[0], id="unexpected"), original[1]]]
        for rows in cases:
            data.write_jsonl(path, rows)
            with self.assertRaises(ValueError):
                self.execute()
        data.write_jsonl(path, original)
        path.write_bytes(path.read_bytes() + b'{"id":')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "incomplete last line"):
            self.execute()
        self.assertEqual(path.read_bytes(), before)

    def test_concurrent_writer_is_rejected(self):
        path = prediction_path("dev", "krikri", "basic_en", "sentence", self.root)
        with run_lock(path), self.assertRaisesRegex(ValueError, "another process"):
            self.execute()

    def test_analysis_rejects_missing_duplicate_or_inconsistent_pairs(self):
        a, _ = self.execute(condition="sentence")
        b, _ = self.execute(condition="context")
        rows_a, rows_b = load_predictions(a), load_predictions(b)
        self.assertEqual(context_effect(rows_a, rows_b)["n"], 2)
        for changed in (rows_b[:1], rows_b + rows_b[:1],
                        [dict(rows_b[0], gold="literal"), rows_b[1]],
                        [dict(rows_b[0], settings={"changed": True}), rows_b[1]]):
            with self.assertRaises(ValueError):
                context_effect(rows_a, changed)
        runs = load_runs("dev", pred_dir=self.root)
        validate_reference(runs, self.examples)
        changed = copy.deepcopy(self.examples)
        changed[0]["context_before"] = "Changed."
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            validate_reference(runs, changed)


class TestReviewRegressions(unittest.TestCase):
    def test_mlx_generation_uses_explicit_greedy_sampler_and_strict_parser(self):
        mlx_lm = types.ModuleType("mlx_lm")
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = Mock(return_value="greedy-sampler")
        mlx_lm.generate = Mock(return_value="not literal")
        model = MLXModel.__new__(MLXModel)
        model.method, model.max_tokens = "generate", 10
        model.model, model.tokenizer = object(), object()
        model._prompt_ids = Mock(return_value=[1, 2])
        with patch.dict("sys.modules", {"mlx_lm": mlx_lm, "mlx_lm.sample_utils": sample_utils}):
            result = model.predict("prompt", {"literal": "literal", "idiomatic": "idiomatic"})
        sample_utils.make_sampler.assert_called_once_with(temp=0.0)
        self.assertEqual(mlx_lm.generate.call_args.kwargs["sampler"], "greedy-sampler")
        self.assertEqual((result["prediction"], result["status"]), (None, "unparsed"))

    def test_negated_partial_and_multiple_answers_are_rejected(self):
        for text in ("not literal", "not idiomatic", "Δεν είναι κυριολεκτικά", "literalness",
                     "literal or idiomatic", "literal\nActually idiomatic.", "", None):
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text))
        for text, label in [(" LITERAL.\n", "literal"), ("μεταφορικά", "idiomatic"),
                            ("ΚΥΡΙΟΛΕΚΤΙΚΆ", "literal"), ("ιδιωματικά", "idiomatic")]:
            self.assertEqual(parse_answer(text), label)

    def test_ambiguous_examples_are_retained_but_excluded_from_inference(self):
        original = {name: data.read_jsonl(path) for name, path in data.SPLITS.items()}
        original["dev"] = data.read_jsonl(data.ROOT / "data/review/pilot/source.jsonl")
        original["dev"][0]["label"] = "ambiguous"
        self.assertEqual(data.check(original["dev"]), [])
        with tempfile.TemporaryDirectory() as folder:
            paths = {name: Path(folder) / f"{name}.jsonl" for name in original}
            for name, rows in original.items():
                data.write_jsonl(paths[name], rows)
            with patch.object(data, "SPLITS", paths):
                self.assertEqual(len(data.load_split("dev")), len(original["dev"]) - 1)
                original["test"][0]["expression_id"] = original["dev"][0]["expression_id"]
                data.write_jsonl(paths["test"], original["test"])
                with self.assertRaisesRegex(SystemExit, "overlap"):
                    data.load_split("dev")
            self.assertEqual(data.read_jsonl(paths["dev"])[0]["label"], "ambiguous")

    def test_empty_spans_are_rejected(self):
        ex = data.load_split("dev")[0]
        self.assertTrue(any("span" in p for p in data.check([dict(ex, span="")])))


class TestFrozenSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        template = Template("basic_en")
        self.choice = {"protocol_version": PROTOCOL_VERSION, "pipeline_hash": pipeline_fingerprint(),
                       "runtime": runtime_versions(),
                       "dev_data_hash": fingerprint(data.load_split("dev")),
                       "models": ["krikri"], "method": "generate", "prompt": template.name,
                       "prompt_hash": template.hash, "model_settings": {"krikri": {"method": "generate"}}}
        self.write_choice()

    def write_choice(self):
        (self.root / "chosen_prompt.json").write_text(json.dumps(self.choice))

    def test_defaults_reuse_saved_models_and_explicit_mismatch_is_rejected(self):
        with patch.object(experiment, "run_all", return_value={}) as run_all, \
                patch.object(experiment, "write_summary"), patch.object(experiment, "print_details"), \
                contextlib.redirect_stdout(io.StringIO()):
            experiment.test(results_dir=self.root)
            self.assertEqual(run_all.call_args.args[1], ["krikri"])
            self.assertEqual(run_all.call_args.kwargs["expected_settings"], self.choice["model_settings"])
            run_all.reset_mock()
            with self.assertRaisesRegex(SystemExit, "models selected on dev"):
                experiment.test(["llama"], results_dir=self.root)
            run_all.assert_not_called()

    def test_stale_data_pipeline_and_prompt_choices_are_rejected_before_inference(self):
        for field in ("dev_data_hash", "pipeline_hash", "prompt_hash"):
            original = self.choice[field]
            self.choice[field] = "changed"
            self.write_choice()
            with patch.object(experiment, "run_all") as run_all, self.assertRaises(SystemExit):
                experiment.test(results_dir=self.root)
            run_all.assert_not_called()
            self.choice[field] = original

    def test_test_command_rejects_dev_only_overrides(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            experiment.main(["test", "--method", "score", "--results-dir", str(self.root)])

    def test_development_and_test_roundtrip_uses_only_requested_runs(self):
        examples = data.load_split("dev")[:2]
        def fake_load(key, method):
            if key == "baseline":
                return AlwaysIdiomatic()
            model = FakeModel()
            model.name = key
            return model
        # A stale, invalid file must not be swept into the new experiment summary.
        unrelated = self.root / "predictions/dev__llama__old__context.jsonl"
        unrelated.parent.mkdir()
        unrelated.write_text('{"unrelated": true}\n')
        with patch.object(experiment, "load_model", side_effect=fake_load), \
                patch.object(experiment, "load_split", return_value=examples), \
                patch.object(experiment, "free_memory"), contextlib.redirect_stdout(io.StringIO()):
            experiment.dev(["krikri"], "generate", ["basic_en"], self.root)
            saved = json.loads((self.root / "chosen_prompt.json").read_text())
            self.assertEqual(saved["models"], ["krikri"])
            self.assertEqual(saved["model_settings"]["krikri"], FakeModel().settings)
            experiment.test(results_dir=self.root)
        self.assertEqual(len(list((self.root / "predictions").glob("test__*.jsonl"))), 4)
        self.assertEqual(unrelated.read_text(), '{"unrelated": true}\n')


if __name__ == "__main__":
    unittest.main()
