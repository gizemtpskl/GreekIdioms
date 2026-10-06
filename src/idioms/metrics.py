"""Accuracy, macro-F1, bootstrap confidence intervals and McNemar's test (stdlib only)."""

import random
from math import comb

from .data import LABELS


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else float("nan")


def macro_f1(gold, pred, labels=LABELS):
    """Unparsed/refused predictions (None) count as wrong for every class."""
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def bootstrap_ci(gold, pred, metric, n_resamples=2000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(gold)
    scores = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(metric([gold[i] for i in idx], [pred[i] for i in idx]))
    scores.sort()
    lo = scores[int(alpha / 2 * n_resamples)]
    hi = scores[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return lo, hi


def paired_bootstrap_diff(gold, pred_a, pred_b, metric, n_resamples=2000, alpha=0.05, seed=0):
    """CI for metric(B) - metric(A) on the same items."""
    rng = random.Random(seed)
    n = len(gold)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        diffs.append(metric(g, [pred_b[i] for i in idx]) - metric(g, [pred_a[i] for i in idx]))
    diffs.sort()
    return diffs[int(alpha / 2 * n_resamples)], diffs[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]


def mcnemar_exact(only_a_correct, only_b_correct):
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def cohen_kappa(labels_a, labels_b):
    n = len(labels_a)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    categories = set(labels_a) | set(labels_b)
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
