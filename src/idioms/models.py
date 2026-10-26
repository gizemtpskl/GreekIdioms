"""The models: an always-idiomatic baseline and two local 8B models run with mlx-lm.

The comparison evaluates a Greek-adapted model and a related general model.
It does not isolate a single training change; tokenizers, instruction tuning,
and quantized builds may also differ.

Two ways to get an answer from a model:
    generate  the model writes a single answer label (greedy decoding)
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

ANSWER_LABELS = {"literal": "literal", "idiomatic": "idiomatic",
                 "κυριολεκτικα": "literal", "μεταφορικα": "idiomatic", "ιδιωματικα": "idiomatic"}


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def parse_answer(text):
    """Accept only one full English/Greek label, optionally followed by a period.

    Case, accents and outer whitespace are ignored. Explanations, negations,
    prefixes, markup and multiple answers are invalid, even on later lines.
    """
    if not isinstance(text, str):
        return None
    normalized = strip_accents(text.strip().casefold())
    match = re.fullmatch(r"(literal|idiomatic|κυριολεκτικα|μεταφορικα|ιδιωματικα)\.?", normalized)
    return ANSWER_LABELS[match.group(1)] if match else None


class AlwaysIdiomatic:
    name = BASELINE
    settings = {}

    def predict(self, prompt, answer_words):
        return {"prediction": "idiomatic", "raw_output": "idiomatic", "status": "ok"}


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
            from mlx_lm.sample_utils import make_sampler

            text = generate(self.model, self.tokenizer, prompt=prompt_ids, max_tokens=self.max_tokens,
                            sampler=make_sampler(temp=0.0))
            prediction = parse_answer(text)
            return {"prediction": prediction, "raw_output": text,
                    "status": "ok" if prediction else "unparsed"}
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
