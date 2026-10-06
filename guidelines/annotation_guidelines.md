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

- The second reviewer labels examples **without seeing** `label`, then writes their label to `label_reviewer2`.
- Discuss disagreements after computing agreement (`python -m idioms.agreement`). Keep the original labels in a copy so agreement can be reported honestly.
