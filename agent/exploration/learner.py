"""Inference engine shared by every explorer.

The hypothesis space is generated from a public WorldSpec.
The inference mechanism is independent of the world's vocabulary.
"""

import math
from itertools import combinations, product

from .environment import (
    NOTHING,
    conj_matches,
)
from .world_spec import CLASSIC_WORLD


def _all_candidates(spec=CLASSIC_WORLD):
    """Generate every possible rule allowed by the world specification."""

    candidates = []

    for interaction in spec.interactions:
        for arity in range(1, spec.max_rule_arity + 1):
            for features in combinations(spec.feature_order, arity):
                value_sets = [
                    spec.features[feature]
                    for feature in features
                ]

                for values in product(*value_sets):
                    conjunction = tuple(zip(features, values))

                    for effect in spec.effects:
                        candidates.append(
                            (interaction, conjunction, effect)
                        )

    return candidates


def _build_interaction_index(candidates):
    """Index candidate rules by interaction for faster lookup."""

    index = {}

    for candidate_index, candidate in enumerate(candidates):
        interaction = candidate[0]

        index.setdefault(interaction, []).append(candidate_index)

    return index


# Backwards-compatible classic-world aliases.
CANDIDATES = _all_candidates(CLASSIC_WORLD)
BY_INTERACTION = _build_interaction_index(CANDIDATES)


class RuleLearner:
    def __init__(self, spec=CLASSIC_WORLD):
        self.spec = spec

        self.candidates = _all_candidates(spec)
        self.by_interaction = _build_interaction_index(
            self.candidates
        )

        self.observations = []

        self.alive = [True] * len(self.candidates)
        self.support = [0] * len(self.candidates)

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
        """Enrich historical observations when new information is discovered."""

        changed = False

        for observation in self.observations:
            if observation["object_id"] != object_id:
                continue

            for feature, value in known_features.items():
                if value is not None:
                    if observation["features"].get(feature) != value:
                        observation["features"][feature] = value
                        changed = True

        if changed:
            self._rebuild()

    def _rebuild(self):
        """Recalculate hypotheses using the complete observation history."""

        self.alive = [True] * len(self.candidates)
        self.support = [0] * len(self.candidates)

        self.n_obs = len(self.observations)
        self.positives = []
        self._chosen = None

        for observation in self.observations:
            interaction = observation["interaction"]
            features = observation["features"]
            outcome = observation["outcome"]

            if outcome != NOTHING:
                self.positives.append(
                    (interaction, dict(features), outcome)
                )

            for index in self.by_interaction[interaction]:
                if not self.alive[index]:
                    continue

                _, conjunction, effect = self.candidates[index]

                if conj_matches(conjunction, features):
                    if effect == outcome:
                        self.support[index] += 1
                    else:
                        self.alive[index] = False

    def testable(self, interaction, features):
        """Count consistent candidate rules tested by an experiment."""

        count = 0

        for index in self.by_interaction[interaction]:
            if (
                self.alive[index]
                and conj_matches(
                    self.candidates[index][1],
                    features,
                )
            ):
                count += 1

        return count

    def info_gain(self, interaction, features):
        """Estimate information gain in bits."""

        total = 0
        counts = {}

        for index in self.by_interaction[interaction]:
            if self.alive[index]:
                total += 1

                candidate = self.candidates[index]

                if conj_matches(candidate[1], features):
                    effect = candidate[2]
                    counts[effect] = counts.get(effect, 0) + 1

        if total == 0:
            return 0.0

        matched = sum(counts.values())

        probabilities = (
            [value / total for value in counts.values()]
            + [(total - matched) / total]
        )

        return -sum(
            probability * math.log2(probability)
            for probability in probabilities
            if probability > 0
        )

    def _select(self):
        """Choose a parsimonious explanation of observed effects."""

        if self._chosen is not None:
            return self._chosen

        chosen = []

        for interaction in self.spec.interactions:
            positives = [
                (features, outcome)
                for observed_interaction, features, outcome
                in self.positives
                if observed_interaction == interaction
            ]

            if not positives:
                continue

            candidates = [
                index
                for index in self.by_interaction[interaction]
                if (
                    self.alive[index]
                    and self.support[index] > 0
                )
            ]

            covers = {}

            for index in candidates:
                _, conjunction, effect = self.candidates[index]

                covers[index] = {
                    position
                    for position, (features, outcome)
                    in enumerate(positives)
                    if (
                        outcome == effect
                        and conj_matches(conjunction, features)
                    )
                }

            uncovered = set(range(len(positives)))

            while uncovered:
                best = None

                for index in candidates:
                    gain = len(covers[index] & uncovered)

                    if gain == 0:
                        continue

                    key = (
                        -gain,
                        len(self.candidates[index][1]),
                        -self.support[index],
                        index,
                    )

                    if best is None or key < best[0]:
                        best = (key, index)

                if best is None:
                    break

                chosen.append(best[1])
                uncovered -= covers[best[1]]

        self._chosen = chosen

        return chosen

    def predict(self, interaction, features):
        """Predict using the simplest matching selected rule."""

        best = None

        for index in self._select():
            candidate_interaction, conjunction, effect = (
                self.candidates[index]
            )

            if (
                candidate_interaction == interaction
                and conj_matches(conjunction, features)
            ):
                key = (len(conjunction), index)

                if best is None or key < best[0]:
                    best = (key, effect)

        return best[1] if best else NOTHING

    def rules(self):
        """Return selected rules."""

        return [
            self.candidates[index]
            for index in self._select()
        ]