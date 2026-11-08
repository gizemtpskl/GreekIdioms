"""The models: an always-idiomatic baseline and local 7-8B models run with mlx-lm.

The comparison evaluates a Greek-adapted model and a related general model.
It does not isolate a single training change; tokenizers, instruction tuning,
and quantized builds may also differ.

Two ways to get an answer from a model:
    generate  the model writes a single answer label (greedy decoding)
    score     the model's log-probability of each of the two answer words is compared

Models without a ready-made 4-bit MLX build are converted once on this machine:

    python -m idioms.models build meltemi
"""

import argparse
import re
import unicodedata

from .data import ROOT

MODELS = {
    "krikri": "ilsp/Llama-Krikri-8B-Instruct-v1.5-4bit-mlx",
    "llama": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "meltemi": "ilsp/Meltemi-7B-Instruct-v1.5",  # full-precision source, converted locally (LOCAL_BUILDS)
}
# the models dev uses when --models is not given
DEFAULT_MODELS = ("krikri", "llama")
# Hugging Face commit of each checkpoint, so every run loads exactly the same weights.
# Change only together with a fresh results directory.
REVISIONS = {
    "krikri": "140ab6af4203751d7f91896e6d534f11a1e1863d",
    "llama": "90215b22ec18e72f623dde2ea7af4097025160e2",
    "meltemi": "77d7fa68ee9f9c0addf6ed93acdc32ebafde9394",
}
# Models converted to 4-bit MLX on this machine, with the same quantization as the
# ready-made builds above. The output folder is not in git; build it with `build`.
LOCAL_BUILDS = {
    "meltemi": {"path": "local_models/Meltemi-7B-Instruct-v1.5-4bit", "q_bits": 4, "q_group_size": 64},
}
BASELINE = "baseline"
METHODS = ("generate", "score")
# score is the main method: the generate output is often a label followed by an explanation,
# which the strict parser counts as unparsed.
DEFAULT_METHOD = "score"

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
    def __init__(self, key, method=DEFAULT_METHOD, max_tokens=10):
        from mlx_lm import load

        if method not in METHODS:
            raise SystemExit(f"method must be one of {METHODS}")
        self.key, self.method, self.max_tokens = key, method, max_tokens
        self.name = key if method == "generate" else f"{key}-score"
        build = LOCAL_BUILDS.get(key)
        if build:
            path = ROOT / build["path"]
            if not (path / "config.json").exists():
                raise SystemExit(f"{key} is not built yet; run: python -m idioms.models build {key}")
            self.model, self.tokenizer = load(str(path))
        else:
            self.model, self.tokenizer = load(MODELS[key], revision=REVISIONS[key])
        self.settings = {"hf_id": MODELS[key], "revision": REVISIONS[key], "method": method,
                         **({"local_build": build} if build else {}),
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


def load_model(key, method=DEFAULT_METHOD):
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


def build_local(key):
    """Download the pinned source checkpoint and convert it to a quantized MLX model."""
    from mlx_lm import convert

    if key not in LOCAL_BUILDS:
        raise SystemExit(f"{key!r} has a ready-made build; nothing to convert. Local builds: {list(LOCAL_BUILDS)}")
    build = LOCAL_BUILDS[key]
    path = ROOT / build["path"]
    if path.exists():
        raise SystemExit(f"{path} already exists; delete it first to rebuild")
    convert(MODELS[key], mlx_path=str(path), quantize=True, q_bits=build["q_bits"],
            q_group_size=build["q_group_size"], revision=REVISIONS[key])
    print(f"{key} built in {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build locally converted models.")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("model", choices=list(LOCAL_BUILDS))
    args = ap.parse_args(argv)
    build_local(args.model)


if __name__ == "__main__":
    main()
