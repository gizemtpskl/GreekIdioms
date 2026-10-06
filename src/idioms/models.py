"""Model backends. Each one takes a prompt and returns (raw_text, status), optionally
followed by a dict of extra fields to save with the prediction.

A backend is chosen with a spec string "<kind>:<name>", e.g.
    baseline:always_idiomatic
    anthropic:claude-opus-5-5
    hf:ilsp/Llama-Krikri-8B-Instruct
    mlx:ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx         (Apple Silicon, no GPU needed)
    mlx-score:ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx   (same model, scores the two answers)
"""

import re
import unicodedata

OK, REFUSED = "ok", "refused"


# Answer words accepted for each label (Greek without accents; any word starting with these stems counts)
ANSWER_STEMS = {
    "literal": ("literal", "κυριολεκτ"),
    "idiomatic": ("idiomatic", "μεταφορικ", "ιδιωματικ"),
}


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def parse_label(text):
    """Map a model answer to 'literal' / 'idiomatic', or None if it isn't exactly one of them.

    Accepts English (literal/idiomatic) and Greek (κυριολεκτικά/μεταφορικά) answers.
    Only the first non-empty line is read, because some models add an explanation
    after the answer (which may mention the other label).
    """
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    words = re.findall(r"\w+", strip_accents(first_line.lower()))
    found = {label for label, stems in ANSWER_STEMS.items() for w in words if w.startswith(stems)}
    return found.pop() if len(found) == 1 else None


class AlwaysIdiomatic:
    """Reference baseline from the proposal: predicts 'idiomatic' for everything."""

    def classify(self, prompt):
        return "idiomatic", OK

    def settings(self):
        return {}


class AnthropicBackend:
    def __init__(self, model, effort="low"):
        import anthropic  # imported lazily so the other backends work without the SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def classify(self, prompt):
        kwargs = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            return "", REFUSED
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, OK

    def settings(self):
        return {"effort": self.effort}


class HFBackend:
    """Local chat model via Hugging Face transformers, greedy decoding."""

    def __init__(self, model_id, max_new_tokens=10):
        from transformers import pipeline

        self.pipe = pipeline("text-generation", model=model_id, device_map="auto")
        self.max_new_tokens = max_new_tokens

    def classify(self, prompt):
        out = self.pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return out[0]["generated_text"][-1]["content"], OK

    def settings(self):
        return {"max_new_tokens": self.max_new_tokens, "do_sample": False}


class MLXBackend:
    """Local model on Apple Silicon via mlx-lm (e.g. ILSP's official 4-bit Krikri), greedy decoding."""

    def __init__(self, model_id, max_tokens=10):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)
        self.max_tokens = max_tokens

    def classify(self, prompt):
        from mlx_lm import generate

        # token ids, not text: encoding the templated text again would add a second start-of-text token
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        return generate(self.model, self.tokenizer, prompt=prompt_ids, max_tokens=self.max_tokens), OK

    def settings(self):
        return {"max_tokens": self.max_tokens, "decoding": "greedy"}


ANSWER_WORDS = {"literal": ("literal", "κυριολεκτικά"), "idiomatic": ("idiomatic", "μεταφορικά")}


def answer_words(prompt):
    """The two answer words the prompt asks for, read from its last line (English or Greek)."""
    last_line = next(line for line in reversed(prompt.splitlines()) if line.strip())
    words = {}
    for label, options in ANSWER_WORDS.items():
        found = [w for w in options if w in last_line]
        if len(found) != 1:
            raise ValueError(f"the prompt's last line must name exactly one answer word for {label!r}: {last_line!r}")
        words[label] = found[0]
    return words


class MLXScoreBackend:
    """Local model on Apple Silicon; instead of generating text, compares how likely
    the model finds each of the two answer words right after the prompt.

    The prediction is the more likely word; the log-probabilities are returned too.
    """

    def __init__(self, model_id):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)

    def word_logprob(self, prompt_ids, word):
        import mlx.core as mx

        word_ids = self.tokenizer.encode(word, add_special_tokens=False)
        ids = prompt_ids + word_ids
        logits = self.model(mx.array([ids]))[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # the token at position i is predicted by the logits at position i - 1
        start = len(prompt_ids)
        return sum(logprobs[i - 1, ids[i]].item() for i in range(start, len(ids)))

    def classify(self, prompt):
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
        words = answer_words(prompt)
        scores = {label: self.word_logprob(prompt_ids, word) for label, word in words.items()}
        best = max(scores, key=scores.get)
        return words[best], OK, {"logprobs": {label: round(s, 4) for label, s in scores.items()}}

    def settings(self):
        return {"decoding": "score answer words"}


def get_backend(spec, effort="low"):
    kind, _, name = spec.partition(":")
    if kind == "baseline" and name == "always_idiomatic":
        return AlwaysIdiomatic()
    if kind == "anthropic" and name:
        return AnthropicBackend(name, effort=effort)
    if kind == "hf" and name:
        return HFBackend(name)
    if kind == "mlx" and name:
        return MLXBackend(name)
    if kind == "mlx-score" and name:
        return MLXScoreBackend(name)
    raise ValueError(f"unknown model spec {spec!r}")
