"""Experiment-selection strategies. Each returns one action dict per call.

They only see what an agent may see: the View (observations so far) and the learner built
from those observations. They never see the environment's hidden rules.
"""
from .environment import CONDITIONS, INTERACTIONS


class View:
    """Everything the explorer has legitimately learned about the world so far."""

    def __init__(self, obs):
        self.objects = [{"shape": o["shape"], "colour": o["colour"], "material": None}
                        for o in obs["objects"]]
        self.cond = (obs["condition"]["temp"], obs["condition"]["light"])
        self.tried = set()  # (object, interaction, (temp, light)) already performed

    def features(self, i, cond=None):
        f = dict(self.objects[i])
        f["temp"], f["light"] = cond or self.cond
        return f


class RandomStrategy:
    name = "random"

    def choose(self, view, learner, rng):
        n = len(view.objects)
        r = rng.random()
        if r < 0.2:
            return {"type": "inspect", "object": rng.randrange(n)}
        if r < 0.4:
            t, l = rng.choice(CONDITIONS)
            return {"type": "set_condition", "temp": t, "light": l}
        return {"type": "interact", "object": rng.randrange(n),
                "interaction": rng.choice(INTERACTIONS)}


class SystematicStrategy:
    """Inspect everything, then sweep: for each condition, every interaction on every object."""
    name = "systematic"

    def __init__(self):
        self.plan = None
        self.pos = 0

    def choose(self, view, learner, rng):
        if self.plan is None:
            n = len(view.objects)
            plan = [{"type": "inspect", "object": i} for i in range(n)]
            for t, l in CONDITIONS:
                plan.append({"type": "set_condition", "temp": t, "light": l})
                for inter in INTERACTIONS:
                    for i in range(n):
                        plan.append({"type": "interact", "object": i, "interaction": inter})
            self.plan = plan
        act = self.plan[self.pos % len(self.plan)]
        self.pos += 1
        return act


class ActiveCountStrategy:
    """Greedy v1: run the untried experiment that puts the most still-consistent candidate rules
    to the test, per unit cost; inspect a new object when little is left to test."""
    name = "active_count"

    def __init__(self, inspect_threshold=20):
        self.inspect_threshold = inspect_threshold

    def score(self, learner, inter, features):
        return learner.testable(inter, features)

    def choose(self, view, learner, rng):
        n = len(view.objects)
        best = None
        for i in range(n):
            if view.objects[i]["material"] is None:
                continue
            for inter in INTERACTIONS:
                for cond in CONDITIONS:
                    if (i, inter, cond) in view.tried:
                        continue
                    cost = 1 if cond == view.cond else 2
                    value = self.score(learner, inter, view.features(i, cond)) / cost
                    key = (value, rng.random())
                    if best is None or key > best[0]:
                        best = (key, i, inter, cond)
        uninspected = [i for i in range(n) if view.objects[i]["material"] is None]
        if uninspected and (best is None or best[0][0] < self.inspect_threshold):
            return {"type": "inspect", "object": uninspected[0]}
        if best is None:
            return {"type": "interact", "object": rng.randrange(n),
                    "interaction": rng.choice(INTERACTIONS)}
        _, i, inter, cond = best
        if cond != view.cond:
            return {"type": "set_condition", "temp": cond[0], "light": cond[1]}
        return {"type": "interact", "object": i, "interaction": inter}


class ActiveStrategy(ActiveCountStrategy):
    """Greedy v2: highest expected information gain (entropy of the outcome over the
    still-consistent candidate rules), per unit cost."""
    name = "active"

    def __init__(self, inspect_threshold=0.5):
        self.inspect_threshold = inspect_threshold

    def score(self, learner, inter, features):
        return learner.info_gain(inter, features)
