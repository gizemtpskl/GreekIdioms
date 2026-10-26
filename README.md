# Understanding Greek Idioms in Context

Can pretrained language models tell literal from idiomatic uses of Modern Greek expressions, and does surrounding context help? Each example is classified twice, once with the **target sentence only** and once **with 1–2 surrounding sentences**. An always-idiomatic baseline gives a reference score.

## Layout

```
data/            annotated examples + splits (see data/README.md for the format)
  sample/        toy examples for testing the pipeline only
  pilot/         pilot set
  final/         full evaluation set
guidelines/      annotation guidelines
prompts/         prompt templates ({expression} and {passage} are filled in)
src/idioms/      pipeline code
  data.py        load / validate / split examples
  prompts.py     build the passage for each condition
  models.py      model backends + answer parsing
  run.py         run one model × condition × split, save predictions
  evaluate.py    accuracy, macro-F1, bootstrap CIs, McNemar, breakdowns
  validate.py    check a data file, show label balance per expression
  agreement.py   inter-annotator agreement (Cohen's kappa)
results/         predictions (one JSONL per run) and metrics.csv
tests/
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[anthropic]"     # Claude models
pip install -e ".[hf]"            # local Hugging Face models (needs a GPU for 7–8B models)
```

For Claude models, set `ANTHROPIC_API_KEY` (or log in with `ant auth login`).

## Workflow

```bash
# 1. check the data
python -m idioms.validate data/pilot/pilot.jsonl --splits data/splits.json
python -m idioms.validate data/final/final.jsonl --splits data/splits.json \
    --against data/pilot/pilot.jsonl

# 2. run a model under both conditions
python -m idioms.run --data data/pilot/pilot.jsonl --splits data/splits.json \
    --split dev --model anthropic:claude-opus-5-5 --condition sentence
python -m idioms.run --data data/pilot/pilot.jsonl --splits data/splits.json \
    --split dev --model anthropic:claude-opus-5-5 --condition context

# baseline
python -m idioms.run --data data/pilot/pilot.jsonl --splits data/splits.json \
    --split dev --model baseline:always_idiomatic --condition sentence

# 3. score
python -m idioms.evaluate results/predictions/dev__*.jsonl --breakdown
python -m idioms.evaluate --compare \
    results/predictions/dev__anthropic-claude-opus-5-5__sentence.jsonl \
    results/predictions/dev__anthropic-claude-opus-5-5__context.jsonl

# 4. agreement on the double-annotated subset
python -m idioms.agreement data/pilot/pilot.jsonl
```

Model specs: `baseline:always_idiomatic`, `anthropic:<model id>`, `hf:<hugging face id>`. Greek-specific open models to consider are ILSP's Krikri and Meltemi (check the exact Hugging Face ids, e.g. `ilsp/Llama-Krikri-8B-Instruct`).

A run that stops halfway can be restarted with the same command; examples already in the output file are skipped. Every prediction records the model, its settings and a hash of the prompt template, so results can be traced back to the setup that produced them.

## Notes on the method

- **Splits are by expression.** Develop prompts on `dev` and run `test` only once the prompt is fixed.
- **Answers that can't be parsed and refusals count as wrong**, and are reported in the `unparsed_or_refused` column. If that column isn't close to zero, fix the prompt on `dev`.
- **Claude settings.** Current Claude models don't accept `temperature`, so repeated runs aren't guaranteed to give identical answers. `--effort` (default `low`) is recorded with each prediction. No automatic model fallback is used, so every prediction in a file comes from the model named in it.
- **Small dataset.** Report the bootstrap confidence intervals, and use McNemar's test for sentence vs. context, because both conditions are run on the same items.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover tests
```
