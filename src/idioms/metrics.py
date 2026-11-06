"""Scores and statistics, standard library only.

A missing prediction (None: the model's answer could not be read) counts as wrong.
"""

import random
from math import comb

from .data import LABELS

N_RESAMPLES = 2000


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else float("nan")


def macro_f1(gold, pred):
    f1s = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1s) / len(f1s)


def _resample(n, seed):
    rng = random.Random(seed)
    for _ in range(N_RESAMPLES):
        yield [rng.randrange(n) for _ in range(n)]


def _interval(values):
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def bootstrap_ci(gold, pred, metric, seed=0):
    """95% percentile bootstrap interval, resampling examples."""
    return _interval(metric([gold[i] for i in idx], [pred[i] for i in idx]) for idx in _resample(len(gold), seed))


def paired_bootstrap_diff(gold, pred_a, pred_b, metric, seed=0):
    """95% interval for metric(B) - metric(A) on the same examples."""
    diffs = []
    for idx in _resample(len(gold), seed):
        g = [gold[i] for i in idx]
        diffs.append(metric(g, [pred_b[i] for i in idx]) - metric(g, [pred_a[i] for i in idx]))
    return _interval(diffs)


def mcnemar_exact(only_a_correct, only_b_correct):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def cohen_kappa(labels_a, labels_b):
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    expected = sum(labels_a.count(c) * labels_b.count(c) for c in set(labels_a) | set(labels_b)) / n**2
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)
