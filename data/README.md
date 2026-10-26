# Data

| file | split | expressions | examples |
|---|---|---|---|
| `pilot/pilot.jsonl` | dev: used only to choose the prompt | e01–e05 | 20 |
| `final/final.jsonl` | test: the reported results | e06–e25 | 100 |

All examples were constructed and labelled by annotator `ann_1` (Angeliki Andreadi). No expression appears in both files.

## Design

Each expression has examples of these kinds:

| kind | label | `sentence_sufficient` | context |
|---|---|---|---|
| clear idiomatic use | idiomatic | `true` | before and after |
| clear literal use | literal | `true` | before and after |
| minimal pair, idiomatic half | idiomatic | `false` | before only |
| minimal pair, literal half | literal | `false` | before only |
| a second idiomatic use (test only) | idiomatic | `true` | before and after |

The two halves of a minimal pair share the same target sentence. Only the preceding context decides the label (`notes`: "context dependent pair").

## Fields (one JSON object per line)

| field | description |
|---|---|
| `id` | unique id (`pilot-001`, `test-001`, …) |
| `expression_id` | id of the expression (`e01` … `e25`) |
| `expression` | the expression in dictionary form, e.g. `σπάω τον πάγο` |
| `target_sentence` | the sentence to classify |
| `span` | the expression **exactly as it appears** in `target_sentence`, including capitals and inflection |
| `context_before` / `context_after` | surrounding sentences (empty string if none) |
| `label` | `literal`, `idiomatic`, or `ambiguous`; ambiguous rows are retained but excluded from binary inference |
| `sentence_sufficient` | `true` if the target sentence alone settles the meaning |
| `source` | where the sentence comes from (`constructed`) |
| `is_constructed` | `true` for sentences written for this dataset |
| `annotator` | who assigned `label` |
| `notes` | free text |
| `label_reviewer2` | *optional:* a second annotator's independent label, for `python -m idioms.agreement` |

`python -m idioms.data` checks that every field is present and of the right type,
required text is nonempty, labels are valid, IDs are unique, and every nonempty
`span` occurs in its sentence. It rejects development/test overlap in example IDs,
expression IDs, or exact expression names. It prints the binary evaluation overview
and the number of excluded ambiguous items. Items marked `review_status: pending`
cannot be used for inference until adjudicated.

The independent-review and adjudication workflow on `ebrar/pipeline-validation`
still needs to be integrated after this branch is merged. Do not replace its
preserved independent labels with final adjudicated labels when calculating agreement.

## Changes since the annotated files were committed

- `pilot.jsonl`, `pilot-020`: removed a stray `;` at the end of the line, which made it invalid JSON.
- `final.jsonl`, `test-010`, `test-075`, `test-076`: `span` capitalised to match the sentence-initial capital letter.
