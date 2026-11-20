"""Inference engine shared by every explorer."""

import math
from itertools import combinations, product

from .environment import (
    EFFECTS, FEATURE_ORDER, FEATURES, INTERACTIONS, NOTHING,
    conj_matches
)


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
        self.observations = []
        self.alive = [True] * len(CANDIDATES)
        self.support = [0] * len(CANDIDATES)
        self.n_obs = 0
        self.positives = []
        self._chosen = None

    def update(self, interaction, features, outcome, object_id=None):
        """Store an observation and rebuild the hypothesis space."""

        self.observations.append({
            "object_id": object_id,
            "interaction": interaction,
            "features": dict(features),
            "outcome": outcome,
        })

        self._rebuild()

    def update_object_features(self, object_id, known_features):
        """Enrich historical observations when new object information is discovered."""

        changed = False

        for obs in self.observations:
            if obs["object_id"] != object_id:
                continue

            for feature, value in known_features.items():
                if value is not None:
                    if obs["features"].get(feature) != value:
                        obs["features"][feature] = value
                        changed = True

        if changed:
            self._rebuild()

    def _rebuild(self):
        """Recalculate all hypotheses using the complete known observation history."""

        self.alive = [True] * len(CANDIDATES)
        self.support = [0] * len(CANDIDATES)
        self.n_obs = len(self.observations)
        self.positives = []
        self._chosen = None

        for obs in self.observations:
            interaction = obs["interaction"]
            features = obs["features"]
            outcome = obs["outcome"]

            if outcome != NOTHING:
                self.positives.append(
                    (interaction, dict(features), outcome)
                )

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
        """Count consistent candidates tested by an experiment."""

        n = 0

        for idx in BY_INTERACTION[interaction]:
            if self.alive[idx] and conj_matches(
                CANDIDATES[idx][1], features
            ):
                n += 1

        return n

    def info_gain(self, interaction, features):
        """Estimate information gain in bits."""

        total = 0
        counts = {}

        for idx in BY_INTERACTION[interaction]:
            if self.alive[idx]:
                total += 1
                candidate = CANDIDATES[idx]

                if conj_matches(candidate[1], features):
                    effect = candidate[2]
                    counts[effect] = counts.get(effect, 0) + 1

        if total == 0:
            return 0.0

        matched = sum(counts.values())

        probs = (
            [v / total for v in counts.values()]
            + [(total - matched) / total]
        )

        return -sum(
            p * math.log2(p)
            for p in probs if p > 0
        )

    def _select(self):
        """Choose a parsimonious explanation of observed effects."""

        if self._chosen is not None:
            return self._chosen

        chosen = []

        for inter in INTERACTIONS:
            pos = [
                (f, o)
                for i2, f, o in self.positives
                if i2 == inter
            ]

            if not pos:
                continue

            cands = [
                idx for idx in BY_INTERACTION[inter]
                if self.alive[idx] and self.support[idx] > 0
            ]

            covers = {}

            for idx in cands:
                _, conj, eff = CANDIDATES[idx]

                covers[idx] = {
                    k for k, (f, o) in enumerate(pos)
                    if o == eff and conj_matches(conj, f)
                }

            uncovered = set(range(len(pos)))

            while uncovered:
                best = None

                for idx in cands:
                    gain = len(covers[idx] & uncovered)

                    if gain == 0:
                        continue

                    key = (
                        -gain,
                        len(CANDIDATES[idx][1]),
                        -self.support[idx],
                        idx
                    )

                    if best is None or key < best[0]:
                        best = (key, idx)

                if best is None:
                    break

                chosen.append(best[1])
                uncovered -= covers[best[1]]

        self._chosen = chosen
        return chosen

    def predict(self, interaction, features):
        """Predict using the simplest matching selected rule."""

        best = None

        for idx in self._select():
            inter, conj, eff = CANDIDATES[idx]

            if inter == interaction and conj_matches(conj, features):
                key = (len(conj), idx)

                if best is None or key < best[0]:
                    best = (key, eff)

        return best[1] if best else NOTHING

    def rules(self):
        """Return selected rules."""

        return [
            CANDIDATES[idx]
            for idx in self._select()
        ]