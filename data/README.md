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
| `adjudicator` / `adjudication_notes` | optional information about who resolved a disagreement and why; may be absent or blank |

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

## Pilot review recorded on 12 November 2026

All 20 pilot examples have a submitted second review. The original submission
in `review/pilot/reviewer.csv` is preserved unchanged. The imported second review
uses the pseudonymous identifier `ann_2`; this identifies the submission, not a
verified name.

Label agreement on the submitted judgments is **17/20 (85%)**, with **Cohen's
kappa = 0.714286**, including the ambiguous judgment. The label disagreements are:

| ID | Primary label | Second reviewer |
|---|---|---|
| `pilot-009` | idiomatic | literal |
| `pilot-012` | literal | ambiguous |
| `pilot-013` | idiomatic | literal |

Sentence-sufficiency agreement is **11/20 (55%)**. Nine other examples differ on
whether the target sentence alone is sufficient: `pilot-003`, `pilot-004`,
`pilot-007`, `pilot-008`, `pilot-011`, `pilot-015`, `pilot-016`, `pilot-019`, and
`pilot-020`. This made **12 cases requiring adjudication** in total, now resolved. Sentence
sufficiency means that the target sentence alone supports the same single
interpretation without the surrounding context. The submitted independent
judgments remain preserved alongside the final decisions.

| File under `review/pilot/` | Purpose |
|---|---|
| `reviewer.csv` | Unmodified submitted review; also preserved in commit `b42aca9` |
| `source.jsonl` | Exact snapshot of the primary pilot annotations used for comparison |
| `reviewer_completed.csv` | Working copy; only `pilot-012`'s spelling `ambigious` is corrected to `ambiguous` |
| `reviewed.jsonl` | Both independent judgments, with 8 agreed and 12 pending examples |
| `agreement.json` | Preserved pre-adjudication snapshot: agreement figures, original pending cases, source hashes, correction record, and reviewer-ID note |
| `adjudication.csv` | Completed final labels and sufficiency decisions uploaded by Angeliki in commit `4250215`; optional attribution and notes are blank |
| `adjudicated.jsonl` | Imported decisions: 12 adjudicated and 8 agreed examples, with both independent judgments preserved |

The importer also applies its existing case/whitespace normalization; it does not
change the judgments. All 12 decisions have now been imported. The final labels
match the primary labels; only `pilot-011` changes sentence sufficiency from
`false` to `true`. There are no pending rows in `adjudicated.jsonl`.
The adjudicated output is now active at `data/pilot/pilot.jsonl`; the original
primary dataset remains in `review/pilot/source.jsonl`. The test review is
described [below](#test-review-recorded-on-16-november-2026).

For any future decision sheet, fill `final_label` and `final_sentence_sufficient`
for every row, including rows where only sufficiency differs. `adjudicator` and
`adjudication_notes` are optional. Each row includes all the reference information;
no second sheet is needed. If a meaning remains unresolved, use `ambiguous` and
set sufficiency to `false`. Leave every reference column unchanged. Neither
reviewer's original answer is automatically preferred.

The completed pilot sheet was imported from the repository root with:

```bash
PYTHONPATH=src python3 -m idioms.review adjudicate data/review/pilot/reviewed.jsonl \
    data/review/pilot/adjudication.csv --out data/review/pilot/adjudicated.jsonl
```

The output already exists; do not rerun this command over it. Keep the independent
agreement figures above for reporting. Adjudication does not
replace them with agreement on the final labels. Adoption and the development-run
reuse check are complete: saved inputs and gold labels are identical, and
`physical_en` remains the selected prompt. See the
[reuse audit](../results/reviewed_dev_check.json) and
[execution limitations](../README.md#checked-development-runs-after-pilot-adjudication).

## Test review recorded on 16 November 2026

`data/review/final/` holds the review of 40 of the 100 test examples:
for each of the 20 test expressions, one literal and one idiomatic example,
drawn with `random.Random(0)` from the IDs sorted within each expression and
label, then shuffled by the export (`--seed 0`). The selection was made before
any test-set model predictions existed. `manifest.json` lists the 40 IDs and
the fingerprint of the test file at export, which is preserved as `source.jsonl`.

The second reviewer (`ann_2`, the same as for the pilot) worked from the blind
`reviewer.csv` only. Agreement with the primary labels is **40/40** (Cohen's
kappa 1.0), and **40/40** on sentence sufficiency (`agreement.json`), so
`adjudication.csv` has no rows and adjudication changed nothing. The review fields
of the 40 examples were added to `final/final.jsonl` without changing any label or
sentence-sufficiency judgment. The remaining 60 examples keep the primary
annotator's labels and are reported as not double-annotated.

## Review workflow

These commands run from the repository root after setup. For annotation work and
tests alone, no model dependencies are needed: use `PYTHONPATH=src python3` in
place of `python` below.

The pilot import below has already been performed. These commands document the
procedure; use new destinations for a new review round rather than overwriting
the recorded artifacts.

1. Read the versioned annotation rules in the
   [guidelines](../guidelines/annotation_guidelines.md) and practice using examples
   outside the review set. Both reviewers should use the same version. Review
   all pilot items. Decide the final-set review subset and
   record its size and selection method before looking at model predictions.
2. Export a blind packet. `data/review/pilot/` contains the existing pilot packet
   generated from the primary pilot (now preserved in `source.jsonl`) with seed 0. Its `reviewer.csv` has now
   been completed; the recorded outputs are described above. The command below shows
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
4. Discuss each row in `adjudication.csv`, which includes the sentence, context,
   both original judgments, reviewer notes, and the type of disagreement. Preserve
   these reference columns and fill `final_label` and `final_sentence_sufficient`.
   `adjudicator` and `adjudication_notes` are optional and can stay blank.
   Save as `adjudication_completed.csv`, then run:

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
   to validate both canonical splits, then freeze the data. Check whether saved
   development predictions remain valid; changed model inputs or gold labels
   require rerunning or rescoring as appropriate. The runner requires matching
   fingerprints and runtime, so reuse across changes needs an explicitly
   documented procedure; otherwise rerun development in a fresh results directory.
   Apply the same review workflow to the test review.

Text corrections belong in reviewer notes during review, not in the passage
columns. If a passage needs changing, preserve the original review history,
revise the source, and collect a new independent review of the changed material.

## Changes since the annotated files were committed

- `pilot.jsonl`, `pilot-020`: removed a stray `;` at the end of the line, which made it invalid JSON.
- `final.jsonl`, `test-010`, `test-075`, `test-076`: `span` capitalised to match the sentence-initial capital letter.
- `final.jsonl`, `test-074`: corrected the English word `theatrical` to Greek `θεατρικής` in the preceding context.
