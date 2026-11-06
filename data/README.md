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
| `label_reviewer1` / `label_reviewer2` | independent labels preserved by the review import, including `ambiguous` |
| `sentence_sufficient_reviewer1` / `sentence_sufficient_reviewer2` | each reviewer's independent judgment of whether context is needed |
| `reviewer2` / `notes_reviewer2` | second reviewer's identifier and notes; explain an ambiguous judgment |
| `review_status` | `agreed`, `pending`, or `adjudicated`; a disagreement in either judgment requires adjudication |
| `adjudicator` / `adjudication_notes` | who resolved a disagreement and why; required after adjudication |

Review fields are absent on unreviewed examples. Once present, the complete
review metadata must be valid. `label` and `sentence_sufficient` remain the
evaluation fields; adjudication can change them without changing either
reviewer's original judgments. A primary `annotator` identifier is required to
export or import a review, and the second reviewer must use a different identifier.

`python -m idioms.data` checks that every field is present and of the right type,
required text is nonempty, labels are valid, IDs are unique, and every nonempty
`span` occurs in its sentence. It rejects development/test overlap in example IDs,
expression IDs, or exact expression names. It prints the binary evaluation overview
and the number of excluded ambiguous items. Items marked `review_status: pending`
cannot be used for inference until adjudicated.

## Review workflow

These commands run from the repository root after setup. For annotation work and
tests alone, no model dependencies are needed: use `PYTHONPATH=src python3` in
place of `python` below.

1. Read the versioned annotation rules in the
   [guidelines](../guidelines/annotation_guidelines.md) and practice using examples
   outside the review set. Both reviewers should use the same version. Review
   all pilot items. Decide the final-set review subset and
   record its size and selection method before looking at model predictions.
2. Export a blind packet. `data/review/pilot/` already contains a blank packet
   generated from the current pilot file with seed 0. The command below shows
   how it was generated; to export again, use a new directory. Commands refuse
   to overwrite existing outputs.

   ```bash
   python -m idioms.review export data/pilot/pilot.jsonl --out-dir data/review/pilot --seed 0
   ```

   `--ids` optionally selects specific example IDs, for example for a predetermined
   final-set subset. `manifest.json` records the selected IDs and a fingerprint
   of the entire source dataset. Give the reviewer only `reviewer.csv` and the
   guidelines. Keep source labels, notes, model predictions, and the rest of the
   repository out of the independent-review handoff. The reviewer fills only
   the three reviewer columns and saves `reviewer_completed.csv` in this folder.
3. Preserve the source before later adopting reviewed data, then import the
   completed CSV. Replace `ann_2` with the actual reviewer's identifier. If the
   source changed after export, import rejects it: reconcile the change and
   export a new packet instead of editing the manifest.

   ```bash
   cp -n data/pilot/pilot.jsonl data/review/pilot/source.jsonl
   python -m idioms.review import data/review/pilot/source.jsonl data/review/pilot/reviewer_completed.csv \
       --manifest data/review/pilot/manifest.json --reviewer ann_2 \
       --out data/review/pilot/reviewed.jsonl --decisions-out data/review/pilot/adjudication.csv
   python -m idioms.agreement data/review/pilot/reviewed.jsonl
   ```

   Record raw agreement, Cohen's kappa, and the reviewed-subset size **before
   discussing disagreements**. Kappa is undefined if both reviewers use only
   one identical category. Agreement always uses preserved independent labels,
   including ambiguous judgments; it never substitutes the final gold labels.
   The agreement command accepts multiple JSONL paths, or defaults to both
   canonical splits if no paths are supplied.
4. Discuss each row in `adjudication.csv`. Preserve its first five columns and
   fill `final_label`, `final_sentence_sufficient`, `adjudicator`, and
   `adjudication_notes`. Save as `adjudication_completed.csv`, then run:

   ```bash
   python -m idioms.review adjudicate data/review/pilot/reviewed.jsonl \
       data/review/pilot/adjudication_completed.csv --out data/review/pilot/adjudicated.jsonl
   ```

   If there are no disagreements, the original header-only `adjudication.csv`
   can be used directly to create the final output. If the meaning remains
   unresolved, use `ambiguous`; such examples stay in the dataset but are excluded
   from binary evaluation. Unreviewed items outside a selected subset keep their
   primary labels, and must be reported as not double-annotated.
5. Inspect and approve the adjudicated output, retain the source and review
   artifacts as history, and copy the approved output to the corresponding
   canonical dataset path (`data/pilot/pilot.jsonl` for this pilot). The experiment
   does not automatically read files in `data/review/`. Run `python -m idioms.data`
   to validate both canonical splits, then freeze the data and rerun development
   in a fresh results directory. Apply the same workflow to the test review.

Text corrections belong in reviewer notes during review, not in the passage
columns. If a passage needs changing, preserve the original review history,
revise the source, and collect a new independent review of the changed material.

## Changes since the annotated files were committed

- `pilot.jsonl`, `pilot-020`: removed a stray `;` at the end of the line, which made it invalid JSON.
- `final.jsonl`, `test-010`, `test-075`, `test-076`: `span` capitalised to match the sentence-initial capital letter.
- `final.jsonl`, `test-074`: corrected the English word `theatrical` to Greek `θεατρικής` in the preceding context.
