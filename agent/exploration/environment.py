"""Stage 1 environment: procedurally generated worlds with hidden rules.

The environment structure is defined by a public WorldSpec.
The hidden objects and rules remain accessible only through truth(),
which is reserved for evaluation.
"""

import random
from itertools import product

from .world_spec import CLASSIC_WORLD


# Backwards-compatible aliases for existing tests and code.
SHAPES = CLASSIC_WORLD.features["shape"]
MATERIALS = CLASSIC_WORLD.features["material"]
COLOURS = CLASSIC_WORLD.features["colour"]
TEMPS = CLASSIC_WORLD.features["temp"]
LIGHTS = CLASSIC_WORLD.features["light"]

FEATURES = CLASSIC_WORLD.features
FEATURE_ORDER = CLASSIC_WORLD.feature_order
OBJECT_FEATURES = CLASSIC_WORLD.object_features

INTERACTIONS = CLASSIC_WORLD.interactions
EFFECTS = CLASSIC_WORLD.effects

NOTHING = "nothing"
CONDITIONS = CLASSIC_WORLD.conditions


class BudgetExhausted(Exception):
    pass


def conj_matches(conj, features):
    """A feature value of None means that the feature is unknown."""
    for feature, value in conj:
        if features.get(feature) != value:
            return False
    return True


def conj_conflict(c1, c2):
    """True if two conjunctions can never both hold."""
    existing = dict(c1)

    for feature, value in c2:
        if feature in existing and existing[feature] != value:
            return True

    return False


def make_conj(pairs, spec=CLASSIC_WORLD):
    """Create a consistently ordered conjunction."""
    return tuple(sorted(
        pairs,
        key=lambda pair: spec.feature_order.index(pair[0])
    ))


class GroundTruth:
    """Hidden objects and rules. Rules are (interaction, conjunction, outcome)."""

    def __init__(self, objects, rules, spec=CLASSIC_WORLD):
        self.objects = objects
        self.rules = rules
        self.spec = spec

    def rule_for(self, interaction, features):
        for rule in self.rules:
            if rule[0] == interaction and conj_matches(rule[1], features):
                return rule

        return None

    def outcome(self, interaction, features):
        rule = self.rule_for(interaction, features)
        return rule[2] if rule else NOTHING

    def space_queries(self):
        """Enumerate the complete feature space without duplicate conditions."""
        out = []

        object_features = self.spec.object_features

        feature_values = [
            self.spec.features[name]
            for name in object_features
        ]

        for values in product(*feature_values):
            base_features = dict(zip(object_features, values))

            for condition in self.spec.conditions:
                features = dict(base_features)

                features.update(dict(zip(
                    self.spec.condition_features,
                    condition,
                )))

                for interaction in self.spec.interactions:
                    out.append({
                        "interaction": interaction,
                        "features": features,
                        "truth": self.outcome(interaction, features),
                    })

        return out

    def queries(self):
        """Enumerate every reachable object/interaction/condition experiment."""
        out = []

        for i, obj in enumerate(self.objects):
            for condition in self.spec.conditions:
                features = dict(obj)

                for name, value in zip(
                    self.spec.condition_features,
                    condition
                ):
                    features[name] = value

                for interaction in self.spec.interactions:
                    rule = self.rule_for(interaction, features)

                    out.append({
                        "id": (i, interaction, *condition),
                        "interaction": interaction,
                        "features": features,
                        "truth": rule[2] if rule else NOTHING,
                        "rule": rule,
                    })

        return out


def _generate_rules(rng, objects, n_rules, spec=CLASSIC_WORLD):
    rules = []
    tries = 0

    while len(rules) < n_rules and tries < 5000:
        tries += 1

        interaction = rng.choice(spec.interactions)
        arity = rng.randint(1, spec.max_rule_arity)

        selected_features = rng.sample(
            spec.feature_order,
            arity
        )

        conjunction = make_conj([
            (feature, rng.choice(spec.features[feature]))
            for feature in selected_features
        ], spec)

        outcome = rng.choice(spec.effects)

        object_part = [
            pair for pair in conjunction
            if pair[0] in spec.object_features
        ]

        if not any(
            conj_matches(object_part, obj)
            for obj in objects
        ):
            continue

        if any(
            rule[0] == interaction
            and not conj_conflict(rule[1], conjunction)
            for rule in rules
        ):
            continue

        rules.append((interaction, conjunction, outcome))

    if len(rules) != n_rules:
        raise RuntimeError(
            f"Could only generate {len(rules)} of {n_rules} requested rules."
        )

    return rules


class Env:
    """A partially observable world governed by a public WorldSpec."""

    def __init__(
        self,
        seed,
        n_objects=8,
        n_rules=3,
        budget=60,
        spec=CLASSIC_WORLD,
    ):
        self.spec = spec

        rng = random.Random(seed)

        objects = [
            {
                feature: rng.choice(spec.features[feature])
                for feature in spec.object_features
            }
            for _ in range(n_objects)
        ]

        rules = _generate_rules(
            rng,
            objects,
            n_rules,
            spec,
        )

        self._truth = GroundTruth(objects, rules, spec)

        self._cond = rng.choice(spec.conditions)
        self._budget = budget
        self._used = 0

    @property
    def budget_left(self):
        return self._budget - self._used

    def truth(self):
        """Evaluator only. Never provide this to the explorer."""
        return self._truth

    def observe(self):
        """Return only publicly observable information."""
        objects = []

        for i, obj in enumerate(self._truth.objects):
            visible = {
                feature: obj[feature]
                for feature in self.spec.observable_features
            }

            objects.append({
                "id": i,
                **visible,
            })

        condition = dict(zip(
            self.spec.condition_features,
            self._cond,
        ))

        return {
            "objects": objects,
            "condition": condition,
            "interactions": list(self.spec.interactions),
            "budget_left": self.budget_left,
        }

    def _spend(self):
        if self._used >= self._budget:
            raise BudgetExhausted()

        self._used += 1

    def _check_object(self, i):
        if (
            not isinstance(i, int)
            or isinstance(i, bool)
            or not 0 <= i < len(self._truth.objects)
        ):
            raise ValueError("invalid object id: %r" % (i,))

    def step(self, action):
        """Validate and execute one action. Invalid actions cost nothing."""
        if not isinstance(action, dict) or "type" not in action:
            raise ValueError("action must be a dict with a 'type'")

        kind = action["type"]

        if kind == "inspect":
            i = action.get("object")
            self._check_object(i)
            self._spend()

            obj = self._truth.objects[i]

            return {
                "id": i,
                **obj,
            }

        if kind == "set_condition":
            condition = tuple(
                action.get(name)
                for name in self.spec.condition_features
            )

            if condition not in self.spec.conditions:
                raise ValueError("invalid condition")

            self._spend()
            self._cond = condition

            return {
                "condition": dict(zip(
                    self.spec.condition_features,
                    condition,
                ))
            }

        if kind == "interact":
            i = action.get("object")
            interaction = action.get("interaction")

            self._check_object(i)

            if interaction not in self.spec.interactions:
                raise ValueError(
                    "invalid interaction: %r" % (interaction,)
                )

            self._spend()

            features = dict(self._truth.objects[i])

            features.update(dict(zip(
                self.spec.condition_features,
                self._cond,
            )))

            return {
                "outcome": self._truth.outcome(
                    interaction,
                    features,
                )
            }

        raise ValueError("unknown action type: %r" % (kind,))