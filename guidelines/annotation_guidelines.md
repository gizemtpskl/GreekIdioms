# Annotation guidelines (draft)

> Draft: the team should revise this after annotating the pilot. Sections marked TODO need decisions.

## 1. Task

For each example, decide how the marked expression is used in the **target sentence**, given its surrounding context:

- **literal**: the words keep their ordinary, compositional meaning (e.g. physically breaking ice).
- **idiomatic**: the expression as a whole has a figurative meaning that can't be built from its parts (e.g. easing social awkwardness).
- **ambiguous**: both readings remain plausible even with the context, or the case is unclear. These examples stay in the file but are left out of the binary evaluation.

## 2. Decision procedure

1. Read the target sentence on its own and write down your reading.
2. Read the context and decide on the final label.
3. Set `sentence_sufficient`:
   - `true` if step 1 already gave you the final label with confidence;
   - `false` if you needed the context.
4. If you hesitate between labels after step 2, use `ambiguous` and explain why in `notes`.

## 3. Matching the expression

- `span` must be copied exactly from the sentence, including inflection (`σπάσει τον πάγο`, `έσπασαν τον πάγο`).
- TODO: Do we include variants with a different possessive or article (`ανοίγω τα χαρτιά μου` vs. `άνοιξε τα χαρτιά του`)? With an inserted word (`έσπασε επιτέλους τον πάγο`)? With changed word order?
- TODO: Do we include passive or nominalised forms (`το σπάσιμο του πάγου`)?

## 4. Hard cases

- **Wordplay / both meanings at once** (e.g. a skater "breaking the ice"): `ambiguous`, plus a note.
- **Metaphor that is not the idiom's meaning**: TODO decide.
- **Quoted or mentioned idioms** (e.g. "the expression 'σπάω τον πάγο' means…"): exclude; don't annotate.

## 5. Selecting examples

- Aim for a roughly equal number of literal and idiomatic examples **for each expression**.
- Prefer natural sentences. If we construct an example, set `is_constructed: true` and say so in `source`.
- Record the exact source (corpus, document, URL) for every natural sentence.
- Context = 1–2 sentences immediately before and/or after the target, copied unchanged.

## 6. Double annotation

Before the pilot review, Angeliki and the reviewer should agree on the open rules
in sections 3 and 4, using examples outside the review set. Record their decisions
here before final annotation. The implementation does not decide these linguistic
questions for them.

- Review all 20 pilot items independently. Select any final-set shared subset
  before inspecting model predictions, and document its size and selection method.
- Give the reviewer only `reviewer.csv` and these guidelines. Do not share the
  original JSONL, primary notes, model predictions, or completed adjudication files
  until their independent review is submitted. Do not tell them the label balance.
- Open the CSV as UTF-8. Preserve its headers, IDs, passages, and row contents.
  Fill only `label_reviewer2`, `sentence_sufficient_reviewer2`, and
  `notes_reviewer2`. Save a completed copy as `reviewer_completed.csv`.
- Read the target first, then the context, following section 2. This CSV shows
  both: sentence sufficiency is a reviewer judgment, not a blinded human
  sentence-only experiment.
- Use `literal`, `idiomatic`, or `ambiguous`; enter `true` or `false` for
  sentence sufficiency. Explain every ambiguous label and any text-quality issue
  in the notes. Flag proposed text corrections there rather than editing passages.
- If both readings remain plausible, or both are intentionally activated, use
  `ambiguous`. Do not force a binary label to match a quota.

After import, compute agreement **before discussing disagreements**. The tool
compares preserved `label_reviewer1` and `label_reviewer2`, including ambiguous
items. Report the shared-subset size, raw agreement, Cohen's kappa, and how the
subset was chosen. If both reviewers use just one identical category, kappa is
undefined; raw agreement is still reported.

Angeliki and the second reviewer then discuss disagreements in either the label
or sentence sufficiency. For each pending item, record `final_label`,
`final_sentence_sufficient`, the adjudicator(s), and a short reason. If the reading
cannot be resolved, set the final label to `ambiguous`; it remains in the dataset
but is excluded from binary evaluation. Do not overwrite independent labels.

The import command creates a new reviewed JSONL and an adjudication CSV. Pending
items cannot be selected for inference. The adjudication command creates another
new JSONL, updating the evaluation `label` and `sentence_sufficient` while retaining
both original judgments. Unreviewed items outside a shared subset keep their
primary labels; report that they were not double-annotated.

If passages change after export, revise the source and export a new review packet.
The import checks the dataset fingerprint and rejects altered passages or a changed
source dataset. Preserve old packets and outputs as the annotation history.
