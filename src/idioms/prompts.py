"""Prompt templates (prompts/<name>.txt) and the passage shown for each condition.

A template contains {expression} and {passage}; its last line names the two
answer words (English literal/idiomatic or Greek κυριολεκτικά/μεταφορικά).
The two conditions use the same template and differ only in the passage:

    sentence:  <target>... <expr>span</expr> ...</target>
    context:   context_before <target>... <expr>span</expr> ...</target> context_after
"""

import hashlib

from .data import ROOT

PROMPT_DIR = ROOT / "prompts"
CONDITIONS = ("sentence", "context")
ANSWER_WORDS = {"literal": ("literal", "κυριολεκτικά"), "idiomatic": ("idiomatic", "μεταφορικά")}


def prompt_names():
    return sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))


class Template:
    def __init__(self, name):
        path = PROMPT_DIR / f"{name}.txt"
        if not path.exists():
            raise SystemExit(f"no prompt {name!r}; available: {', '.join(prompt_names())}")
        self.name = name
        self.text = path.read_text(encoding="utf-8")
        self.hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]
        self.answer_words = self._answer_words()

    def _answer_words(self):
        last_line = next(line for line in reversed(self.text.splitlines()) if line.strip())
        words = {}
        for label, options in ANSWER_WORDS.items():
            found = [w for w in options if w in last_line]
            if len(found) != 1:
                raise SystemExit(f"prompt {self.name!r}: its last line must name one answer word for {label!r}")
            words[label] = found[0]
        return words

    def fill(self, example, condition):
        return self.text.format(expression=example["expression"], passage=passage(example, condition))


def passage(example, condition):
    sentence = example["target_sentence"].replace(example["span"], f"<expr>{example['span']}</expr>", 1)
    target = f"<target>{sentence}</target>"
    if condition == "sentence":
        return target
    if condition == "context":
        parts = [example["context_before"], target, example["context_after"]]
        return " ".join(p.strip() for p in parts if p.strip())
    raise ValueError(f"condition must be one of {CONDITIONS}")
