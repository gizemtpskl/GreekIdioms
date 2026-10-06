"""Model backends. Each one takes a prompt and returns (raw_text, status).

A backend is chosen with a spec string "<kind>:<name>", e.g.
    baseline:always_idiomatic
    anthropic:claude-opus-5-5
    hf:ilsp/Llama-Krikri-8B-Instruct
"""

import re

OK, REFUSED = "ok", "refused"


def parse_label(text):
    """Map a model answer to 'literal' / 'idiomatic', or None if it isn't exactly one of them."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    found = [label for label in ("literal", "idiomatic") if label in words]
    return found[0] if len(found) == 1 else None


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


def get_backend(spec, effort="low"):
    kind, _, name = spec.partition(":")
    if kind == "baseline" and name == "always_idiomatic":
        return AlwaysIdiomatic()
    if kind == "anthropic" and name:
        return AnthropicBackend(name, effort=effort)
    if kind == "hf" and name:
        return HFBackend(name)
    raise ValueError(f"unknown model spec {spec!r}")
