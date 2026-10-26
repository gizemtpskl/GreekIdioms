# Understanding Greek Idioms in Context

Can pretrained language models tell literal from idiomatic uses of Modern Greek expressions, and does surrounding context help? Every example is classified twice: with the **target sentence only**, and **with the surrounding sentences**. An always-idiomatic baseline gives a reference score.

## Data

| split | file | expressions | examples | use |
|---|---|---|---|---|
| dev | `data/pilot/pilot.jsonl` | 5 | 20 (10 literal / 10 idiomatic) | choosing the prompt |
| test | `data/final/final.jsonl` | 20 | 100 (40 literal / 60 idiomatic) | the reported results |

No expression appears in both splits. Every expression includes a **minimal pair**: the same target sentence twice, literal in one example and idiomatic in the other, distinguishable only by its context. See [data/README.md](data/README.md) for the fields. `python -m idioms.data` checks both files and prints an overview.

## Models

| name | model | why |
|---|---|---|
| `krikri` | [ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx) | Llama-3.1-8B with continued pretraining on Greek (ILSP) |
| `llama` | [mlx-community/Llama-3.1-8B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.1-8B-Instruct-4bit) | the same base model without the Greek training |
| `baseline` | — | always answers "idiomatic" |

The two models differ only in Krikri's Greek training, so comparing them isolates its effect. Both are 4-bit builds that run on a 16 GB Apple Silicon Mac with [mlx-lm](https://github.com/ml-explore/mlx-lm) (each about 4.6 GB, downloaded on first use). Answers are produced by greedy generation and read from the first line (`--method generate`, default), or by comparing the log-probabilities of the two answer words (`--method score`).

## Prompts

All prompts are in `prompts/`. `{expression}` and `{passage}` are filled in, and the last line names the two answer words. In the passage, the target sentence is wrapped in `<target>` tags and the expression in `<expr>` tags. The context condition adds the surrounding sentences around it.

| prompt | language | approach |
|---|---|---|
| `basic_en` / `basic_el` | English / Greek | literal vs. idiomatic, with a short definition of each |
| `use_en` / `use_el` | English / Greek | says plainly to judge *this* use, not whether the phrase is an idiom in general |
| `physical_el` | Greek | does the phrase describe something that really happens? |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover tests
```

## Running the experiment

```bash
python -m idioms.experiment dev     # every prompt x both models x both conditions on dev; chooses the prompt
python -m idioms.experiment test    # the chosen prompt only, on test; prints the full analysis
python -m idioms.figures --split test --prompt <chosen prompt>
```

- **The dev step** saves its choice in `results/chosen_prompt.json`: the prompt with the highest macro-F1, averaged over both models and both conditions.
- **The test step** can only use that prompt, and stops if the prompt file was edited after it was chosen. This prevents tuning on test data.
- **Interrupted runs can be restarted:** finished predictions are kept, so rerunning a command continues where it stopped.
- **Close memory-heavy apps first:** each model needs about 5 GB.

Outputs:

| file | content |
|---|---|
| `results/predictions/<split>__<model>__<prompt>__<condition>.jsonl` | one prediction per example, with the raw model output, settings and a fingerprint of the prompt |
| `results/<split>_summary.csv` | accuracy and macro-F1 with 95% bootstrap CIs per run |
| `results/figures/` | scores, accuracy by example type, context effect, per-expression accuracy (PNG + PDF) |

Other commands:

```bash
python -m idioms.run --split dev --model krikri --prompt use_en --condition context   # a single run
python -m idioms.evaluate --split test --prompt use_en --details                       # analysis tables
python -m idioms.agreement                                                             # Cohen's kappa, once label_reviewer2 exists
```

## Analysis

- **Accuracy and macro-F1** with 95% bootstrap confidence intervals. On test, macro-F1 is the main measure: 60% of the examples are idiomatic, so always answering "idiomatic" already reaches 0.60 accuracy (macro-F1 0.375).
- **Context effect:** accuracy with context minus without, with a paired bootstrap CI, the numbers of errors fixed and introduced, and McNemar's exact test.
- **By example type:** accuracy on sentence-sufficient vs. context-dependent examples (`sentence_sufficient`), and the share of minimal pairs where both examples are right. Without context, a model sees the same input for both halves of a pair, so it can get at most one of them right.
- **Per expression:** accuracy for each expression in each condition.

## Layout

```
data/        pilot.jsonl (dev), final.jsonl (test), data/README.md
prompts/     prompt templates
src/idioms/  data.py, prompts.py, models.py, run.py, metrics.py, evaluate.py,
             experiment.py, figures.py, agreement.py
results/     predictions, summaries, figures
tests/
```
