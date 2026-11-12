# Understanding Greek Idioms in Context

Can pretrained language models tell literal from idiomatic uses of Modern Greek expressions, and does surrounding context help? Every example is classified twice: with the **target sentence only**, and **with the surrounding sentences**. An always-idiomatic baseline gives a reference score.

## Data

| split | file | expressions | examples | use |
|---|---|---|---|---|
| dev | `data/pilot/pilot.jsonl` | 5 | 20 (10 literal / 10 idiomatic) | choosing the prompt |
| test | `data/final/final.jsonl` | 20 | 100 (40 literal / 60 idiomatic) | the reported results |

No expression appears in both splits. Every expression currently includes a **minimal pair**: the same target sentence twice, with different reference labels and contexts. These constructed examples still require independent Greek review. See [data/README.md](data/README.md) for the fields. `python -m idioms.data` checks both files, rejects split overlap, and prints an overview. Ambiguous examples remain in the files and are excluded from binary evaluation.

## Models

| name | model | why |
|---|---|---|
| `krikri` | [ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx) | Llama-3.1-8B with continued pretraining on Greek (ILSP) |
| `llama` | [mlx-community/Llama-3.1-8B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.1-8B-Instruct-4bit) | the same base model without the Greek training |
| `meltemi` | [ilsp/Meltemi-7B-Instruct-v1.5](https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5), converted to 4-bit locally | Mistral-7B with continued pretraining on Greek (ILSP); candidate, not in the default model list |
| `baseline` | — | always answers "idiomatic" |

Meltemi has no ready-made 4-bit MLX build. Convert it once (downloads about 15 GB, writes about 4 GB to `local_models/`, which is not in git): `python -m idioms.models build meltemi`. Then pass it explicitly, e.g. `--models krikri llama meltemi`.

These models provide a comparison between a Greek-adapted model and a related general model. It does not isolate a single training change: tokenization, instruction tuning, and quantized builds may also differ. Both use [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon, loaded at pinned checkpoint revisions. The main method compares the log-probabilities of the two answer words (`--method score`, default), so every example gets one of the two labels. Its raw log-probabilities are not calibrated confidence scores. Greedy generation (`--method generate`) is available as a separate setting, but in trials the models usually add an explanation after the label, which the strict parser counts as `unparsed` (see below).

## Prompts

All prompts are in `prompts/`. `{expression}` and `{passage}` are filled in, and the last line names the two answer words. In the passage, the target sentence is wrapped in `<target>` tags and the expression in `<expr>` tags. The context condition adds the surrounding sentences around it.

| prompt | language | approach |
|---|---|---|
| `basic_en` / `basic_el` | English / Greek | literal vs. idiomatic, with a short definition of each |
| `use_en` / `use_el` | English / Greek | says plainly to judge *this* use, not whether the phrase is an idiom in general |
| `physical_en` / `physical_el` | English / Greek | short question after the passage: does the phrase describe something that really happens? |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m unittest discover tests
```

## Independent annotation review

The review tools export a shuffled CSV without the first annotator's labels or
notes, import a second review, and record adjudication while preserving both
independent judgments. See [the workflow commands](data/README.md#review-workflow)
and [annotation guidelines](guidelines/annotation_guidelines.md).

The completed pilot review and imported judgments are in `data/review/pilot/`.
Label agreement is 17/20 (85%), with Cohen's kappa 0.714286. Twelve examples need
adjudication: three label disagreements and nine additional sentence-sufficiency
disagreements. See the [recorded review and handoff](data/README.md#pilot-review-recorded-on-12-november-2026).
The original submission is preserved, and final decisions remain blank.

For a new independent review, read the annotation rules and practice on examples
outside the review set, then give the reviewer only a blank CSV and the guidelines.
Compute agreement before discussing disagreements. Resolve all pending labels
and sentence-sufficiency judgments before using reviewed data for inference;
ambiguous examples remain excluded from binary evaluation.

The experiment reads the canonical `data/pilot/pilot.jsonl` and
`data/final/final.jsonl` files. Review commands create separate outputs; after
human review, retain the annotation history and adopt the approved output at the
corresponding canonical path before freezing the experiment. The canonical pilot
still contains the primary annotations; the reviewed copy is awaiting adjudication.

## Running the experiment

```bash
python -m idioms.experiment dev --results-dir results/verified
# After dataset review and freezing the experimental setup:
python -m idioms.experiment test --results-dir results/verified
python -m idioms.figures --split test --prompt <chosen prompt> --results-dir results/verified
```

- **The dev step** saves its choice in `<results-dir>/chosen_prompt.json`: the prompt with the highest macro-F1, averaged over the selected models and both conditions. It also records the method, model settings, development-data fingerprint, and inference-code fingerprint.
- **The test step** reuses the saved models, settings, method and prompt. It rejects a different `--models` list or changed development data, inference code, or prompt. `--method` and `--prompts` are dev-only options.
- **Interrupted runs can be restarted:** only verified completed predictions are skipped. Changed inputs, labels, settings, prompts, inference code, or runtime versions require a new results directory.
- **Close memory-heavy apps first:** each model needs about 5 GB.

Outputs:

| file | content |
|---|---|
| `results/predictions/<split>__<model>__<prompt>__<condition>.jsonl` | one prediction per example, with the raw model output, settings and a fingerprint of the prompt |
| matching `*.manifest.json` | full run identity, expected items, input fingerprints, prompt text, settings and runtime versions |
| `results/predictions/failures/` | failed inference calls; separate from classification results |
| `results/<split>_summary.csv` | accuracy and macro-F1 with 95% bootstrap CIs per run |
| `results/figures/` | scores, accuracy by example type, context effect, per-expression accuracy (PNG + PDF) |

Other commands:

```bash
python -m idioms.run --split dev --model krikri --prompt use_en --condition context --out-dir results/verified/predictions
python -m idioms.evaluate --split test --prompt use_en --details --results-dir results/verified
python -m idioms.agreement data/review/pilot/reviewed.jsonl  # before adjudication
```

The output paths above are relative to `--results-dir` (default `results`), or
`--out-dir` for a single run. The checked-in `results/predictions/` and
`results/chosen_prompt.json` are historical pilot trials from the earlier pipeline.
They are preserved unchanged, but cannot be resumed or used to start final testing.
Use a fresh results directory to rerun development with the corrected pipeline.
To inspect the old scores explicitly:

```bash
python -m idioms.evaluate --split dev --allow-legacy
```

Legacy mode cannot verify the original inputs or parser version. It does not
upgrade old records or make them eligible for resumption. It also cannot mix
legacy predictions with new manifested predictions in the same comparison.

### Answer parsing and failures

Generated answers must consist of a single label: `literal`, `idiomatic`,
`κυριολεκτικά`, `μεταφορικά`, or `ιδιωματικά`. Case, accents, outer whitespace,
and one final period are tolerated. Explanations, markup, negation and additional
lines are `unparsed`, with a null prediction that counts as wrong. Refusals, when
explicitly reported by a backend, also have null predictions. Neither is retried
to obtain a better answer. Earlier generation trials used a more permissive parser.

A local inference exception is logged as `request_error` and stops the run; it
does not produce a prediction or enter the accuracy denominator. Resolve the
underlying error and rerun the same command to continue. Automatic retries are
not used for local memory/configuration errors. The current models run locally;
there is no API backend in this branch.

Evaluation rejects incomplete new runs, duplicate IDs, mismatched reference labels,
and incompatible paired runs. A damaged last JSONL line requires inspection before
resumption. Each output file allows one writer at a time. Each model is loaded
at a pinned Hugging Face commit (`REVISIONS` in `src/idioms/models.py`), and the
repository ID and commit are recorded in every manifest.

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
             review.py
results/     predictions, summaries, figures
tests/
```
