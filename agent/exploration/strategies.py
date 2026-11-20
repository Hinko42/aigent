"""Experiment-selection strategies.

Strategies operate only on the View and RuleLearner.
They never receive access to hidden environment rules.
"""

from .world_spec import CLASSIC_WORLD


class View:
    """Everything the explorer has legitimately learned about the world."""

    def __init__(self, obs, spec=CLASSIC_WORLD):
        self.spec = spec

        self.objects = []

        for observed_object in obs["objects"]:
            obj = {
                feature: observed_object.get(feature)
                for feature in spec.object_features
            }

            self.objects.append(obj)

        self.cond = tuple(
            obs["condition"][name]
            for name in spec.condition_features
        )

        # (object, interaction, condition) already performed.
        self.tried = set()

    def features(self, i, cond=None):
        """Combine known object features with the selected condition."""

        features = dict(self.objects[i])

        condition = cond if cond is not None else self.cond

        features.update(dict(zip(
            self.spec.condition_features,
            condition,
        )))

        return features


class RandomStrategy:
    name = "random"

    def choose(self, view, learner, rng):
        n = len(view.objects)
        spec = view.spec

        r = rng.random()

        if r < 0.2:
            return {
                "type": "inspect",
                "object": rng.randrange(n),
            }

        if r < 0.4:
            condition = rng.choice(spec.conditions)

            action = {"type": "set_condition"}
            action.update(dict(zip(
                spec.condition_features,
                condition,
            )))

            return action

        return {
            "type": "interact",
            "object": rng.randrange(n),
            "interaction": rng.choice(spec.interactions),
        }


class SystematicStrategy:
    """Inspect everything, then systematically sweep the experiment space."""

    name = "systematic"

    def __init__(self):
        self.plan = None
        self.pos = 0

    def choose(self, view, learner, rng):
        if self.plan is None:
            n = len(view.objects)
            spec = view.spec

            plan = [
                {"type": "inspect", "object": i}
                for i in range(n)
            ]

            for condition in spec.conditions:
                action = {"type": "set_condition"}
                action.update(dict(zip(
                    spec.condition_features,
                    condition,
                )))

                plan.append(action)

                for interaction in spec.interactions:
                    for i in range(n):
                        plan.append({
                            "type": "interact",
                            "object": i,
                            "interaction": interaction,
                        })

            self.plan = plan

        action = self.plan[self.pos % len(self.plan)]
        self.pos += 1

        return action


class ActiveCountStrategy:
    """Choose experiments that test the most surviving hypotheses per cost."""

    name = "active_count"

    def __init__(self, inspect_threshold=20):
        self.inspect_threshold = inspect_threshold

    def score(self, learner, interaction, features):
        return learner.testable(interaction, features)

    def choose(self, view, learner, rng):
        n = len(view.objects)
        spec = view.spec

        best = None

        for i in range(n):
            if any(
                view.objects[i].get(feature) is None
                for feature in spec.object_features
            ):
                continue

            for interaction in spec.interactions:
                for condition in spec.conditions:
                    if (i, interaction, condition) in view.tried:
                        continue

                    cost = 1 if condition == view.cond else 2

                    value = (
                        self.score(
                            learner,
                            interaction,
                            view.features(i, condition),
                        )
                        / cost
                    )

                    key = (value, rng.random())

                    if best is None or key > best[0]:
                        best = (key, i, interaction, condition)

        uninspected = [
            i
            for i in range(n)
            if any(
                view.objects[i].get(feature) is None
                for feature in spec.object_features
            )
        ]

        if uninspected and (
            best is None
            or best[0][0] < self.inspect_threshold
        ):
            return {
                "type": "inspect",
                "object": uninspected[0],
            }

        if best is None:
            return {
                "type": "interact",
                "object": rng.randrange(n),
                "interaction": rng.choice(spec.interactions),
            }

        _, i, interaction, condition = best

        if condition != view.cond:
            action = {"type": "set_condition"}
            action.update(dict(zip(
                spec.condition_features,
                condition,
            )))

            return action

        return {
            "type": "interact",
            "object": i,
            "interaction": interaction,
        }


class ActiveStrategy(ActiveCountStrategy):
    """Choose experiments by expected information gain per cost."""

    name = "active"

    def __init__(self, inspect_threshold=0.5):
        self.inspect_threshold = inspect_threshold

    def score(self, learner, interaction, features):
        return learner.info_gain(interaction, features)