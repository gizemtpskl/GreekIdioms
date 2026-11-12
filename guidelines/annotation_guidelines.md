# Annotation guidelines

> Review protocol v1.0 — 6 November 2026. Apply these rules consistently to both
> primary and independent annotation. These are project scope decisions, not a
> claim that particular dataset examples have already passed linguistic review.

## 1. Task

For each example, decide how the marked expression is used in the **target sentence**, given its surrounding context:

- **literal**: the marked words have their ordinary, compositional meaning in this passage.
- **idiomatic**: this use conveys a conventional figurative meaning of the specified expression. The meaning can be partly transparent; it does not have to be impossible to guess from the words.
- **ambiguous**: both readings remain plausible after context, both are intentionally activated, or there is not enough evidence for a reliable binary judgment. Explain why. These examples remain in the file but are excluded from binary evaluation.

Judge this occurrence, not whether the expression can be an idiom in general.
An unusual or physically unlikely event is not automatically idiomatic: a literal
event can be fictional, hypothetical, or negated. Conversely, figurative language
elsewhere in a sentence does not determine the label of the marked expression.

The review sheet has only these three labels. Section 4 also explains how to
flag out-of-scope examples without misclassifying them as literal or idiomatic.

## 2. Decision procedure

1. Read the target sentence and marked expression before consulting the context.
   Form a provisional reading; do not consult another item's context to decide.
2. Read the supplied preceding and following context. Ask whether this use has
   an ordinary compositional meaning, a conventional idiomatic meaning, or no
   single supported interpretation. Do not invent missing context.
3. Enter your label in `label_reviewer2`. If both readings remain plausible, use
   `ambiguous` and explain why in `notes_reviewer2`.
4. Set `sentence_sufficient_reviewer2` to `true` only if the target sentence alone
   supported the same single binary interpretation confidently. Use `false` if
   context was needed, changed the interpretation, or left the case unresolved.
   For any `ambiguous` review label, enter `false` under this convention.
5. Record uncertainty or text problems in `notes_reviewer2`. Do not edit the
   supplied sentence, context, expression, or span.

The primary annotator follows the same procedure using `label`,
`sentence_sufficient`, and `notes`. Sentence sufficiency is a judgment about the
information available, not a separate confidence score.

## 3. Matching the expression

These rules determine whether an occurrence is an eligible realization of the
specified expression. They do not determine its literal/idiomatic label.

| Form | Rule for this evaluation |
|---|---|
| Inflection | Include ordinary changes of person, number, tense, aspect, and mood when the same expression remains identifiable. |
| Possessives, pronouns, articles | Include grammatical changes that preserve the expression's identity and participant roles; a different possessor or article is not itself evidence for either label. |
| Inserted words | Include modifiers or adverbs when the expression remains recognizable. Include intervening words in the marked span. |
| Word order | Include grammatical reordering when the same lexical expression and relations remain identifiable. |
| Lexical substitution | Do not assume that a synonym creates an equivalent idiom. Include only an established variant identified during expression selection; flag an unrecognized replacement for review. |
| Passive forms and nominalizations | Exclude from this first evaluation to keep the variant policy manageable. If one reaches the review sheet, flag it using section 4. A noun form in a dictionary-style `expression` entry does not make a nominalized target occurrence eligible. |

For dataset preparation, `span` must be one exact, contiguous substring of the
target sentence, preserving capitalization and inflection. Include the words
needed to identify the occurrence and any words between them; do not replace
them with the dictionary form. For example, an inserted adverb belongs inside
the span rather than being silently removed. If the marked span identifies the
wrong words or leaves the target unclear, flag it in notes rather than repairing
the review sheet.

## 4. Hard cases

- **Both readings remain plausible:** use `ambiguous` and describe the two
  readings. The existence of a possible literal reading in a different imagined
  situation is not sufficient; both must be plausible in the supplied passage.
- **Wordplay deliberately activates both readings:** use `ambiguous`, even if
  the wordplay itself is clear. Explain the dual use.
- **Figurative but not idiomatic:** `idiomatic` requires a conventional meaning
  of the specified expression, not just any metaphor. A clearly novel metaphor,
  simile, or other figurative use outside its established senses is outside the
  binary task. Enter `ambiguous` with a note beginning `other_figurative:` and
  explain the interpretation. Do not label it literal merely because it is not
  the intended idiom.
- **Mention rather than use:** a sentence discussing an expression as a phrase
  or giving its definition is outside scope. Enter `ambiguous` with an
  `out_of_scope:` note. Quotation marks alone do not exclude an item: ordinary
  reported speech can still use an expression literally or idiomatically.
- **Excluded form or wrong expression:** flag a passive, nominalized, or
  unrecognized expression form as `ambiguous` with an `out_of_scope:` note.
- **Missing context, unfamiliar meaning, or unclear text:** do not guess. Use
  `ambiguous` and explain what prevents a reliable decision. For a material text
  or span problem, start the note with `text_issue:`. A harmless typo that does
  not affect the interpretation may be noted alongside a binary label.

The `other_figurative:`, `out_of_scope:`, and `text_issue:` notes distinguish
review flags from semantic ambiguity; they are not new labels. Preserve every
row in the returned CSV. During adjudication, reconsider flagged cases using
the same rules. If still ineligible or unresolved, retain `ambiguous` so the
pipeline excludes the item. Report exclusion reasons separately rather than
describing every excluded item as having two possible meanings.

### Practice illustrations

These constructed English examples explain the decision procedure without
revealing labels for any Greek review item. They are not evaluation data.

| Expression and situation | Decision |
|---|---|
| `spill the beans`: a bag tears and dried beans fall onto the kitchen floor | `literal`: ordinary spilling of beans |
| `spill the beans`: someone reveals the secret location of a surprise party | `idiomatic`: the conventional meaning is revealing a secret |
| `spill the beans`: a cook spills actual beans while revealing a secret, and the passage explicitly jokes about both meanings | `ambiguous`: both meanings are deliberately active |
| `spill the beans`: a passage provides too little information to tell whether beans fell or a secret was revealed | `ambiguous`: insufficient context to choose |

In a new poetic image unrelated to a conventional idiomatic meaning, apply the
`other_figurative:` rule instead of assuming that all figurative language is an
instance of the idiom.

## 5. Selecting examples

- Label according to the passage, never an expected balance of labels. Do not force an unnatural literal reading to fill a quota.
- Prefer natural sentences. If we construct an example, set `is_constructed: true` and say so in `source`.
- Record the exact source (corpus, document, URL) for every natural sentence.
- Context = 1–2 sentences immediately before and/or after the target, copied unchanged.

## 6. Double annotation

Before the pilot review, Angeliki and the reviewer should read this protocol and
practice its decision procedure on examples outside the review set. Discuss the
rules, not the labels of review items. Use the same version for both reviewers;
record any later rule change and apply it consistently to affected annotations.

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
or sentence sufficiency. Use `adjudication.csv`: each row contains the sentence,
context, both independent judgments, and reviewer notes. For each pending item,
fill `final_label` and `final_sentence_sufficient`. The `adjudicator` and
`adjudication_notes` columns are optional and may remain blank. Leave the
reference columns unchanged. If the reading
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
