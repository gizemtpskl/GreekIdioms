import unittest

from idioms.data import check, load_split, minimal_pairs
from idioms.metrics import accuracy, cohen_kappa, macro_f1, mcnemar_exact
from idioms.models import DEFAULT_METHOD, DEFAULT_MODELS, LOCAL_BUILDS, MODELS, REVISIONS, parse_answer
from idioms.prompts import Template, passage, prompt_names
from idioms.evaluate import context_effect, grid_rows, pair_accuracy

EXAMPLE = {
    "id": "x-1", "expression_id": "e01", "expression": "σπάω τον πάγο",
    "target_sentence": "Κάποιος έπρεπε να σπάσει τον πάγο.", "span": "σπάσει τον πάγο",
    "context_before": "Η λίμνη είχε παγώσει.", "context_after": "", "label": "literal",
    "sentence_sufficient": False, "source": "constructed", "is_constructed": True,
}


class TestData(unittest.TestCase):
    def test_real_files_are_valid(self):
        dev, test = load_split("dev"), load_split("test")
        self.assertEqual((len(dev), len(test)), (20, 100))
        self.assertFalse({e["expression"] for e in dev} & {e["expression"] for e in test})

    def test_minimal_pairs_in_real_data(self):
        self.assertEqual(len(minimal_pairs(load_split("dev"))), 5)
        self.assertEqual(len(minimal_pairs(load_split("test"))), 20)

    def test_check_finds_problems(self):
        self.assertEqual(check([EXAMPLE]), [])
        bad = dict(EXAMPLE, span="έσπασε τον πάγο", label="")
        problems = check([bad, bad])
        self.assertTrue(any("span" in p for p in problems))
        self.assertTrue(any("label" in p for p in problems))
        self.assertTrue(any("duplicate" in p for p in problems))


class TestPrompts(unittest.TestCase):
    def test_conditions_differ_only_by_context(self):
        sentence, context = passage(EXAMPLE, "sentence"), passage(EXAMPLE, "context")
        self.assertEqual(sentence, "<target>Κάποιος έπρεπε να <expr>σπάσει τον πάγο</expr>.</target>")
        self.assertEqual(context, "Η λίμνη είχε παγώσει. " + sentence)

    def test_every_prompt_fills_and_names_its_answers(self):
        for name in prompt_names():
            t = Template(name)
            text = t.fill(EXAMPLE, "context")
            self.assertNotIn("{", text, name)
            self.assertIn("<expr>", text, name)
            self.assertEqual(set(t.answer_words), {"literal", "idiomatic"}, name)


class TestModels(unittest.TestCase):
    def test_every_model_has_a_pinned_commit(self):
        self.assertEqual(set(REVISIONS), set(MODELS))
        for revision in REVISIONS.values():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_default_and_local_models_are_known(self):
        self.assertTrue(set(DEFAULT_MODELS) <= set(MODELS))
        self.assertTrue(set(LOCAL_BUILDS) <= set(MODELS))

    def test_score_is_the_main_method(self):
        self.assertEqual(DEFAULT_METHOD, "score")


class TestAnswers(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_answer("literal"), "literal")
        self.assertEqual(parse_answer("  ΚΥΡΙΟΛΕΚΤΙΚΆ.\n"), "literal")
        self.assertIsNone(parse_answer("Idiomatic.\n\nThe phrase is not literal"))
        self.assertIsNone(parse_answer("**Απάντηση: ΚΥΡΙΟΛΕΚΤΙΚΆ**"))
        self.assertEqual(parse_answer("μεταφορικά."), "idiomatic")
        self.assertIsNone(parse_answer("λιττερά"))
        self.assertIsNone(parse_answer("literal or idiomatic"))
        self.assertIsNone(parse_answer(""))


class TestMetrics(unittest.TestCase):
    def test_scores(self):
        gold = ["literal", "literal", "idiomatic", "idiomatic"]
        self.assertEqual(macro_f1(gold, gold), 1.0)
        self.assertEqual(accuracy(gold, ["idiomatic"] * 4), 0.5)
        self.assertAlmostEqual(macro_f1(gold, ["idiomatic"] * 4), (0 + 2 / 3) / 2)
        self.assertEqual(accuracy(["literal"], [None]), 0.0)

    def test_statistics(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertAlmostEqual(cohen_kappa(list("aabb"), list("abab")), 0.0)

    def test_context_effect_and_pairs(self):
        def rec(i, gold, pred):
            return {"id": i, "gold": gold, "prediction": pred}
        sentence = [rec("a", "literal", "idiomatic"), rec("b", "idiomatic", "idiomatic"), rec("c", "literal", "literal")]
        context = [rec("a", "literal", "literal"), rec("b", "idiomatic", "idiomatic"), rec("c", "literal", "idiomatic")]
        effect = context_effect(sentence, context)
        self.assertEqual((effect["fixed"], effect["broken"]), (["a"], ["c"]))
        self.assertEqual(pair_accuracy(context, [("a", "b"), ("c", "b")]), 0.5)


class TestGrid(unittest.TestCase):
    def test_grid_has_one_row_per_model_and_one_column_per_prompt(self):
        def recs(*preds):
            return [{"gold": g, "prediction": p} for g, p in zip(("literal", "idiomatic"), preds)]
        runs = {("baseline", "a", "sentence"): recs("idiomatic", "idiomatic"),
                ("baseline", "a", "context"): recs("idiomatic", "idiomatic"),
                ("krikri", "a", "sentence"): recs("literal", "idiomatic"),
                ("krikri", "b", "context"): recs("literal", "literal")}
        prompts, rows = grid_rows(runs)
        self.assertEqual(prompts, ["a", "b"])
        self.assertEqual([r["model"] for r in rows], ["baseline", "krikri"])
        self.assertEqual(rows[0]["b"], rows[0]["a"])  # baseline is the same under every prompt
        self.assertEqual(rows[1]["a"], "1.00 / -")
        self.assertEqual(rows[1]["b"], "- / 0.33")


if __name__ == "__main__":
    unittest.main()
