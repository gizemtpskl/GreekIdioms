"""The models: an always-idiomatic baseline and two local 8B models run with mlx-lm.

Krikri is Llama-3.1-8B with continued pretraining on Greek, so comparing the two
isolates the effect of the Greek training. Both are ILSP / Meta 4-bit builds
that fit on a 16 GB Apple Silicon Mac.

Two ways to get an answer from a model:
    generate  the model writes its answer (greedy decoding); the first line is parsed
    score     the model's log-probability of each of the two answer words is compared
"""

import re
import unicodedata

MODELS = {
    "krikri": "ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx",
    "llama": "mlx-community/Llama-3.1-8B-Instruct-4bit",
}
BASELINE = "baseline"
METHODS = ("generate", "score")

# answer stems, accent-free: any word starting with one of them counts
ANSWER_STEMS = {"literal": ("literal", "κυριολεκτ"), "idiomatic": ("idiomatic", "μεταφορικ", "ιδιωματικ")}


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def parse_answer(text):
    """'literal' / 'idiomatic' from a generated answer, or None if it doesn't name exactly one.
    Only the first non-empty line counts: models sometimes add an explanation after it."""
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    words = re.findall(r"\w+", strip_accents(first_line.lower()))
    found = {label for label, stems in ANSWER_STEMS.items() for w in words if w.startswith(stems)}
    return found.pop() if len(found) == 1 else None


class AlwaysIdiomatic:
    name = BASELINE
    settings = {}

    def predict(self, prompt, answer_words):
        return {"prediction": "idiomatic", "raw_output": ""}


class MLXModel:
    def __init__(self, key, method="generate", max_tokens=10):
        from mlx_lm import load

        if method not in METHODS:
            raise SystemExit(f"method must be one of {METHODS}")
        self.key, self.method, self.max_tokens = key, method, max_tokens
        self.name = key if method == "generate" else f"{key}-score"
        self.model, self.tokenizer = load(MODELS[key])
        self.settings = {"hf_id": MODELS[key], "method": method,
                         **({"max_tokens": max_tokens, "decoding": "greedy"} if method == "generate" else {})}

    def _prompt_ids(self, prompt):
        # token ids straight from the chat template (encoding its text again would add a second <|begin_of_text|>)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )

    def predict(self, prompt, answer_words):
        prompt_ids = self._prompt_ids(prompt)
        if self.method == "generate":
            from mlx_lm import generate

            text = generate(self.model, self.tokenizer, prompt=prompt_ids, max_tokens=self.max_tokens)
            return {"prediction": parse_answer(text), "raw_output": text}
        logprobs = {label: self._logprob(prompt_ids, word) for label, word in answer_words.items()}
        best = max(logprobs, key=logprobs.get)
        return {"prediction": best, "raw_output": answer_words[best],
                "logprobs": {label: round(lp, 4) for label, lp in logprobs.items()}}

    def _logprob(self, prompt_ids, word):
        """Log-probability of the whole answer word right after the prompt (sum over its tokens)."""
        import mlx.core as mx

        ids = prompt_ids + self.tokenizer.encode(word, add_special_tokens=False)
        logits = self.model(mx.array([ids]))[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sum(logprobs[i - 1, ids[i]].item() for i in range(len(prompt_ids), len(ids)))


def load_model(key, method="generate"):
    if key == BASELINE:
        return AlwaysIdiomatic()
    if key not in MODELS:
        raise SystemExit(f"model must be one of {[BASELINE, *MODELS]}, got {key!r}")
    return MLXModel(key, method)


def free_memory():
    """Release a model's memory before loading the next one (needed on 16 GB machines)."""
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass
