import unittest
from pathlib import Path

from idioms.data import load_jsonl, load_splits, select, validate
from idioms.metrics import accuracy, cohen_kappa, macro_f1, mcnemar_exact
from idioms.models import parse_label
from idioms.prompts import build_passage, build_prompt, load_template

ROOT = Path(__file__).resolve().parent.parent
TOY = ROOT / "data/sample/toy.jsonl"


class TestParseLabel(unittest.TestCase):
    def test_clean_answers(self):
        self.assertEqual(parse_label("literal"), "literal")
        self.assertEqual(parse_label("Idiomatic."), "idiomatic")

    def test_unclear_answers(self):
        self.assertIsNone(parse_label("It could be literal or idiomatic"))
        self.assertIsNone(parse_label("κυριολεκτικά"))
        self.assertIsNone(parse_label(""))


class TestData(unittest.TestCase):
    def test_toy_file_is_valid(self):
        self.assertEqual(validate(load_jsonl(TOY)), [])

    def test_span_must_occur_in_sentence(self):
        ex = dict(load_jsonl(TOY)[0], span="έσπασε τον πάγο")
        self.assertTrue(any("does not occur" in p for p in validate([ex])))

    def test_select_drops_ambiguous_and_respects_split(self):
        examples = load_jsonl(TOY)
        splits = load_splits(ROOT / "data/sample/splits.json")
        test = select(examples, splits, "test")
        self.assertEqual({ex["expression_id"] for ex in test}, {"e02"})
        self.assertNotIn("ambiguous", {ex["label"] for ex in test})


class TestPrompts(unittest.TestCase):
    def test_conditions_differ_only_by_context(self):
        ex = load_jsonl(TOY)[0]
        sentence = build_passage(ex, "sentence")
        context = build_passage(ex, "context")
        self.assertIn("<expr>σπάσει τον πάγο</expr>", sentence)
        self.assertIn(sentence, context)
        self.assertIn(ex["context_before"], context)
        self.assertNotIn(ex["context_before"], sentence)

    def test_template_fills(self):
        template = load_template(ROOT / "prompts/classify.txt")
        prompt = build_prompt(template, load_jsonl(TOY)[0], "context")
        self.assertIn("σπάω τον πάγο", prompt)
        self.assertNotIn("{passage}", prompt)


class TestMetrics(unittest.TestCase):
    def test_accuracy_and_f1(self):
        gold = ["literal", "literal", "idiomatic", "idiomatic"]
        self.assertEqual(accuracy(gold, gold), 1.0)
        self.assertEqual(macro_f1(gold, gold), 1.0)
        always_idiomatic = ["idiomatic"] * 4
        self.assertEqual(accuracy(gold, always_idiomatic), 0.5)
        self.assertAlmostEqual(macro_f1(gold, always_idiomatic), (0 + 2 / 3) / 2)

    def test_unparsed_counts_as_wrong(self):
        self.assertEqual(accuracy(["literal"], [None]), 0.0)

    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)

    def test_kappa(self):
        self.assertEqual(cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)


if __name__ == "__main__":
    unittest.main()
