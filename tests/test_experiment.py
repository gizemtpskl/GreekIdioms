import unittest

from idioms.experiment import BASELINE, choose_prompt, main


class TestExperiment(unittest.TestCase):
    def test_prompt_ranking_ignores_baseline_and_averages(self):
        rows = [
            {"model": "m1", "prompt": "a", "macro_f1": 0.6},
            {"model": "m2", "prompt": "a", "macro_f1": 0.8},
            {"model": "m1", "prompt": "b", "macro_f1": 0.9},
            {"model": "m2", "prompt": "b", "macro_f1": 0.3},
            {"model": BASELINE, "prompt": "b", "macro_f1": 1.0},
        ]
        ranking = choose_prompt(rows, ["prompts/a.txt", "prompts/b.txt"])
        self.assertEqual([name for _, name, _ in ranking], ["a", "b"])
        self.assertAlmostEqual(ranking[0][0], 0.7)

    def test_test_split_allows_only_one_prompt(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--data", "data/sample/toy.jsonl", "--splits", "data/sample/splits.json",
                  "--split", "test", "--prompts", "prompts/use_en.txt", "prompts/use_el.txt"])
        self.assertIn("only the prompt chosen on dev", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
