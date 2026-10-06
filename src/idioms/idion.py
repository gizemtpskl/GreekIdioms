"""Turn IDION entries into draft examples for annotation.

    python -m idioms.idion 1 2 3 4 5 --out data/pilot/examples.jsonl

For each IDION entry (https://greek.idion.athenarc.gr/lemma/view/<id>) the
usage examples IDION marks as "acceptable" become draft rows. IDION's
"unacceptable" examples are invented by its documentators and are skipped;
acceptable examples with source "introspection" are invented too and are
marked `is_constructed: true`.

Drafts leave `label` and `span` empty and have no context: the annotators fill
these in (the source URL is kept so the surrounding sentences can be looked up).
`python -m idioms.validate` lists what is still missing.

IDION content is CC-BY-NC; cite Markantonatou et al. (2019, 2020).
"""

import argparse
import html
import re
import time
import urllib.request
from pathlib import Path

from .data import write_jsonl

BASE_URL = "https://greek.idion.athenarc.gr/lemma/view/{}"
USER_AGENT = "GreekIdioms student project (literal vs. idiomatic evaluation)"


def fetch(entry_id):
    request = urllib.request.Request(BASE_URL.format(entry_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def text_of(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_example(li):
    """Split one <li> into (sentence, judgment, source). Judgment is None if it can't be found."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", li))
    text = re.split(r"#\s*sent_id\s*=", text)[0]  # some examples have a UD analysis appended
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"^(.*)\s(acceptable|unacceptable)\s+(?:\d+\s+)?(.*)$", text)
    if not match:
        return text, None, ""
    return match.group(1).strip(), match.group(2), match.group(3).strip()


def parse_entry(page):
    headword = re.search(r"<h1>(.*?)</h1>", page, re.S)
    if not headword:
        raise ValueError("no headword found (entry missing or page layout changed)")
    meaning_el = re.search(r"<h2>Meaning</h2>\s*<h3>(.*?)</h3>", page, re.S)
    meaning_en = re.search(r"<h4>([^<]*)</h4>\s*<p>English</p>", page)
    section = re.search(r"<h2>Corpus Examples</h2>\s*<ul>(.*?)</ul>", page, re.S)
    examples = [parse_example(li) for li in re.findall(r"<li>(.*?)</li>", section.group(1), re.S)] if section else []
    variants = [text_of(v) for v in headword.group(1).split("\n") if text_of(v)]  # one variant per line
    return {
        "headword": variants[0],
        "variants": variants,
        "meaning_el": text_of(meaning_el.group(1)) if meaning_el else "",
        "meaning_en": text_of(meaning_en.group(1)) if meaning_en else "",
        "examples": examples,
    }


def draft_rows(entry_id, entry):
    rows = []
    for sentence, judgment, source in entry["examples"]:
        if judgment == "unacceptable":
            continue
        rows.append({
            "id": f"idion{entry_id}-{len(rows) + 1:02d}",
            "expression_id": f"idion{entry_id}",
            "expression": entry["headword"],
            "target_sentence": sentence,
            "span": "",
            "context_before": "",
            "context_after": "",
            "label": "",
            "sentence_sufficient": None,
            "source": source,
            "is_constructed": source == "introspection",  # IDION's label for examples its team made up
            "annotator": "",
            "notes": "" if judgment else "CHECK: IDION judgment not found, sentence may contain extra text",
            "idion_entry": entry_id,
            "idion_variants": entry["variants"],
            "idion_meaning_el": entry["meaning_el"],
            "idion_meaning_en": entry["meaning_en"],
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entry_ids", nargs="+", type=int, help="IDION entry numbers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite the output file")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; it may contain annotations. Use --force to overwrite.")

    all_rows = []
    for i, entry_id in enumerate(args.entry_ids):
        if i:
            time.sleep(1)  # be gentle with the IDION server
        entry = parse_entry(fetch(entry_id))
        rows = draft_rows(entry_id, entry)
        skipped = len(entry["examples"]) - len(rows)
        print(f"{entry_id}: {entry['headword']} | {entry['meaning_en'] or entry['meaning_el']} | {len(rows)} examples ({skipped} unacceptable skipped)")
        all_rows.extend(rows)

    write_jsonl(out, all_rows)
    print(f"\n{len(all_rows)} draft examples written to {out}")


if __name__ == "__main__":
    main()
