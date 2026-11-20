"""Inference engine shared by every explorer.

It keeps every candidate rule (interaction, 1-2 feature conjunction, outcome) that is still
consistent with the observations, and how many observations support each one. It never sees
the hidden rules, only (interaction, features, outcome) triples. Because all explorers use
the same inference, differences between them come from their choice of experiments.
"""
import math
from itertools import combinations, product

from .environment import (EFFECTS, FEATURE_ORDER, FEATURES, INTERACTIONS, NOTHING,
                          conj_matches)


def _all_candidates():
    cands = []
    for inter in INTERACTIONS:
        for k in (1, 2):
            for feats in combinations(FEATURE_ORDER, k):
                for vals in product(*[FEATURES[f] for f in feats]):
                    conj = tuple(zip(feats, vals))
                    for eff in EFFECTS:
                        cands.append((inter, conj, eff))
    return cands


CANDIDATES = _all_candidates()
BY_INTERACTION = {}
for _idx, _c in enumerate(CANDIDATES):
    BY_INTERACTION.setdefault(_c[0], []).append(_idx)


class RuleLearner:
    def __init__(self):
        self.alive = [True] * len(CANDIDATES)
        self.support = [0] * len(CANDIDATES)
        self.n_obs = 0
        self.positives = []   # (interaction, features, outcome) for every observed effect
        self._chosen = None   # cached parsimonious explanation, reset on every update

    def update(self, interaction, features, outcome):
        """A candidate is refuted if it matches the known features but predicts another outcome.
        Candidates that mention an unknown feature (None) cannot match, so are left alone."""
        self.n_obs += 1
        self._chosen = None
        if outcome != NOTHING:
            self.positives.append((interaction, dict(features), outcome))
        for idx in BY_INTERACTION[interaction]:
            if not self.alive[idx]:
                continue
            _, conj, eff = CANDIDATES[idx]
            if conj_matches(conj, features):
                if eff == outcome:
                    self.support[idx] += 1
                else:
                    self.alive[idx] = False

    def testable(self, interaction, features):
        """How many still-consistent candidates this experiment would put to the test."""
        n = 0
        for idx in BY_INTERACTION[interaction]:
            if self.alive[idx] and conj_matches(CANDIDATES[idx][1], features):
                n += 1
        return n

    def info_gain(self, interaction, features):
        """Entropy (bits) of the outcome of this experiment if each still-consistent candidate
        were equally likely to be the true rule. Higher = splits the candidates more evenly."""
        total = 0
        counts = {}
        for idx in BY_INTERACTION[interaction]:
            if self.alive[idx]:
                total += 1
                c = CANDIDATES[idx]
                if conj_matches(c[1], features):
                    counts[c[2]] = counts.get(c[2], 0) + 1
        if total == 0:
            return 0.0
        matched = sum(counts.values())
        probs = [v / total for v in counts.values()] + [(total - matched) / total]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _select(self):
        """Parsimonious explanation: per interaction, greedily choose the consistent, supported
        candidates that cover the most not-yet-explained observed effects, simplest first."""
        if self._chosen is not None:
            return self._chosen
        chosen = []
        for inter in INTERACTIONS:
            pos = [(f, o) for (i2, f, o) in self.positives if i2 == inter]
            if not pos:
                continue
            cands = [idx for idx in BY_INTERACTION[inter]
                     if self.alive[idx] and self.support[idx] > 0]
            covers = {}
            for idx in cands:
                _, conj, eff = CANDIDATES[idx]
                covers[idx] = {k for k, (f, o) in enumerate(pos)
                               if o == eff and conj_matches(conj, f)}
            uncovered = set(range(len(pos)))
            while uncovered:
                best = None
                for idx in cands:
                    gain = len(covers[idx] & uncovered)
                    if gain == 0:
                        continue
                    key = (-gain, len(CANDIDATES[idx][1]), -self.support[idx], idx)
                    if best is None or key < best[0]:
                        best = (key, idx)
                if best is None:
                    break
                chosen.append(best[1])
                uncovered -= covers[best[1]]
        self._chosen = chosen
        return chosen

    def predict(self, interaction, features):
        """Outcome of the simplest chosen rule that matches; otherwise 'nothing'."""
        best = None
        for idx in self._select():
            inter, conj, eff = CANDIDATES[idx]
            if inter == interaction and conj_matches(conj, features):
                key = (len(conj), idx)
                if best is None or key < best[0]:
                    best = (key, eff)
        return best[1] if best else NOTHING

    def rules(self):
        """The chosen explanation as (interaction, conj, outcome) rules."""
        return [CANDIDATES[idx] for idx in self._select()]
