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
  idion.py       turn IDION entries into draft examples for annotation
results/         predictions (one JSONL per run) and metrics.csv
tests/
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[anthropic]"     # Claude models
pip install -e ".[hf]"            # local Hugging Face models (needs a GPU for 7–8B models)
pip install -e ".[mlx]"           # local models on an Apple Silicon Mac, no GPU needed
```

For Claude models, set `ANTHROPIC_API_KEY` (or log in with `ant auth login`).

## Workflow

```bash
# 0. draft examples from IDION entries (by entry number), then annotate them by hand
python -m idioms.idion 1 2 3 4 5 --out data/pilot/examples.jsonl

# 1. check the data
python -m idioms.validate data/pilot/examples.jsonl --splits data/pilot/splits.json

# 2. run a model under both conditions
python -m idioms.run --data data/pilot/examples.jsonl --splits data/pilot/splits.json \
    --split dev --model anthropic:claude-opus-5-5 --condition sentence
python -m idioms.run --data data/pilot/examples.jsonl --splits data/pilot/splits.json \
    --split dev --model anthropic:claude-opus-5-5 --condition context

# baseline
python -m idioms.run --data data/pilot/examples.jsonl --splits data/pilot/splits.json \
    --split dev --model baseline:always_idiomatic --condition sentence

# 3. score
python -m idioms.evaluate results/predictions/dev__*.jsonl --breakdown
python -m idioms.evaluate --compare \
    results/predictions/dev__anthropic-claude-opus-5-5__sentence.jsonl \
    results/predictions/dev__anthropic-claude-opus-5-5__context.jsonl

# 4. agreement on the double-annotated subset
python -m idioms.agreement data/pilot/examples.jsonl
```

Model specs: `baseline:always_idiomatic`, `anthropic:<model id>`, `hf:<hugging face id>`, `mlx:<hugging face id>`. The two models compared are:

- **Krikri** (Greek-adapted): `ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx`, ILSP's official 4-bit build (4.6 GB).
- **Llama 3.1 8B Instruct** (not adapted): `mlx-community/Llama-3.1-8B-Instruct-4bit` (4.5 GB).

Krikri is Llama-3.1-8B with continued training on Greek text and Greek words added to its vocabulary, so the comparison isolates the effect of that Greek training. Both run on a 16 GB Apple Silicon Mac in about 1.5–5 s per example, e.g. `--model mlx:ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx`.

`mlx-score:<id>` runs the same local model without generating text. It compares how likely the model finds each of the two answer words named on the prompt's last line, so there are no explanations or unreadable answers. Each prediction also stores `logprobs` (a confidence score per label) for later analysis. It takes about 2–4 s per example.

A run that stops halfway can be restarted with the same command; examples already in the output file are skipped. Every prediction records the model, its settings and a hash of the prompt template, so results can be traced back to the setup that produced them.

## Prompts

| file | language | approach |
|---|---|---|
| `classify.txt` | English | first version: literal vs. idiomatic, short definitions |
| `classify_el.txt` | Greek | Greek translation of `classify.txt` |
| `use_en.txt` | English | says plainly to judge *this* use, not whether the phrase is an idiom in general |
| `use_el.txt` | Greek | Greek translation of `use_en.txt`, answer in Greek |
| `physical_el.txt` | Greek | short question: does the phrase describe something that really happens? |

Choose the prompt on the **dev** split, using macro-F1 averaged over both conditions, and fix it before running on test. Run all prompts with `--prompt prompts/<file>`; the prompt name is part of the output filename and appears as a column in `evaluate`. Never edit a prompt that already has results. Save the change as a new file instead (`run` refuses to mix versions).

## Notes on the method

- **Splits are by expression.** Develop prompts on `dev` and run `test` only once the prompt is fixed.
- **Answers that can't be parsed and refusals count as wrong**, and are reported in the `unparsed_or_refused` column. If that column isn't close to zero, fix the prompt on `dev`.
- **Claude settings.** Current Claude models don't accept `temperature`, so repeated runs aren't guaranteed to give identical answers. `--effort` (default `low`) is recorded with each prediction. No automatic model fallback is used, so every prediction in a file comes from the model named in it.
- **Small dataset.** Report the bootstrap confidence intervals, and use McNemar's test for sentence vs. context, because both conditions are run on the same items.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover tests
```
