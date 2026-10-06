# Data

- `sample/` contains toy examples we wrote **only for testing the pipeline**. They are not part of the dataset and must not be reported as results.
- `pilot/` is for the pilot set (about 5 expressions, 10–20 examples).
- `final/` is for the expanded set (about 20 expressions, 120–160 examples).

Each folder holds an `examples.jsonl` file and a `splits.json` file.

## Example format (`examples.jsonl`, one JSON object per line)

| field | required | description |
|---|---|---|
| `id` | yes | unique id, e.g. `pilot-001` |
| `expression_id` | yes | id of the expression, e.g. `e01`; splits are defined over these ids |
| `expression` | yes | dictionary (citation) form, e.g. `σπάω τον πάγο` |
| `target_sentence` | yes | the sentence to classify |
| `span` | yes | the expression **exactly as it appears** in `target_sentence` (inflected form), e.g. `σπάσει τον πάγο` |
| `context_before` | no | 1–2 sentences before the target (empty string if none) |
| `context_after` | no | 1–2 sentences after the target (empty string if none) |
| `label` | yes | `literal`, `idiomatic` or `ambiguous` (ambiguous examples are kept in the file but left out of the evaluation) |
| `sentence_sufficient` | recommended | `true` if the target sentence alone settles the meaning, `false` if you need the context to decide |
| `source` | yes | where the sentence comes from (corpus + document/URL), or how it was constructed |
| `is_constructed` | yes | `true` for examples we wrote ourselves, `false` for natural examples |
| `annotator` | recommended | who assigned `label` |
| `label_reviewer2` | optional | second reviewer's independent label, used by `python -m idioms.agreement` |
| `notes` | optional | anything worth remembering (difficult case, reason for ambiguity, …) |

## Splits (`splits.json`)

```json
{"dev": ["e01", "e02"], "test": ["e03", "e04", "..."]}
```

Splits are made by **expression**, not by example, so every example of an expression is in the same split. Use `dev` to develop the prompt. Run on `test` only after the prompt is fixed.
