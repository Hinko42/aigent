"""Stage 1 environment: a small procedurally generated world with hidden rules.

Standard library only. The agent-facing interface is Env.observe() and Env.step().
Env.truth() is for the evaluator only and must never be handed to an explorer.
"""
import random

SHAPES = ("cube", "sphere", "cone")
MATERIALS = ("wood", "metal", "glass")
COLOURS = ("red", "blue", "green")
TEMPS = ("cold", "hot")
LIGHTS = ("dim", "bright")

FEATURES = {"shape": SHAPES, "material": MATERIALS, "colour": COLOURS,
            "temp": TEMPS, "light": LIGHTS}
FEATURE_ORDER = ("shape", "material", "colour", "temp", "light")
OBJECT_FEATURES = ("shape", "material", "colour")

INTERACTIONS = ("heat", "strike", "soak", "shine")
EFFECTS = ("melts", "breaks", "glows", "fizzes", "darkens")
NOTHING = "nothing"
CONDITIONS = tuple((t, l) for t in TEMPS for l in LIGHTS)


class BudgetExhausted(Exception):
    pass


def conj_matches(conj, features):
    """conj is a tuple of (feature, value) pairs; a feature value of None means unknown."""
    for f, v in conj:
        if features.get(f) != v:
            return False
    return True


def conj_conflict(c1, c2):
    """True if the two conjunctions contradict each other (can never both hold)."""
    d = dict(c1)
    for f, v in c2:
        if f in d and d[f] != v:
            return True
    return False


def make_conj(pairs):
    return tuple(sorted(pairs, key=lambda p: FEATURE_ORDER.index(p[0])))


class GroundTruth:
    """The hidden objects and rules. Rules are (interaction, conj, outcome) tuples."""

    def __init__(self, objects, rules):
        self.objects = objects
        self.rules = rules

    def rule_for(self, interaction, features):
        for r in self.rules:
            if r[0] == interaction and conj_matches(r[1], features):
                return r
        return None

    def outcome(self, interaction, features):
        r = self.rule_for(interaction, features)
        return r[2] if r else NOTHING

    def space_queries(self):
        """Every combination in the whole attribute space, independent of the objects present
        and of what any explorer did. Used to test whether the rules themselves were learned."""
        out = []
        for shape in SHAPES:
            for material in MATERIALS:
                for colour in COLOURS:
                    for temp, light in CONDITIONS:
                        feats = {"shape": shape, "material": material, "colour": colour,
                                 "temp": temp, "light": light}
                        for inter in INTERACTIONS:
                            out.append({"interaction": inter, "features": feats,
                                        "truth": self.outcome(inter, feats)})
        return out

    def queries(self):
        """Every reachable experiment: each object x interaction x condition."""
        out = []
        for i, obj in enumerate(self.objects):
            for temp, light in CONDITIONS:
                feats = dict(obj)
                feats["temp"] = temp
                feats["light"] = light
                for inter in INTERACTIONS:
                    rule = self.rule_for(inter, feats)
                    out.append({"id": (i, inter, temp, light), "interaction": inter,
                                "features": feats,
                                "truth": rule[2] if rule else NOTHING,
                                "rule": rule})
        return out


def _generate_rules(rng, objects, n_rules):
    rules = []
    tries = 0
    while len(rules) < n_rules and tries < 5000:
        tries += 1
        inter = rng.choice(INTERACTIONS)
        k = rng.choice((1, 2))
        feats = rng.sample(FEATURE_ORDER, k)
        conj = make_conj([(f, rng.choice(FEATURES[f])) for f in feats])
        outcome = rng.choice(EFFECTS)
        obj_part = [p for p in conj if p[0] in OBJECT_FEATURES]
        if not any(conj_matches(obj_part, o) for o in objects):
            continue  # must be reachable with the objects that exist
        if any(r[0] == inter and not conj_conflict(r[1], conj) for r in rules):
            continue  # rules of one interaction must not overlap
        rules.append((inter, conj, outcome))
    return rules


class Env:
    """Partially observable: observe() shows shape and colour, inspect() reveals material."""

    def __init__(self, seed, n_objects=8, n_rules=3, budget=60):
        rng = random.Random(seed)
        objects = [{"shape": rng.choice(SHAPES), "material": rng.choice(MATERIALS),
                    "colour": rng.choice(COLOURS)} for _ in range(n_objects)]
        rules = _generate_rules(rng, objects, n_rules)
        self._truth = GroundTruth(objects, rules)
        self._cond = rng.choice(CONDITIONS)
        self._budget = budget
        self._used = 0

    @property
    def budget_left(self):
        return self._budget - self._used

    def truth(self):
        """Evaluator only."""
        return self._truth

    def observe(self):
        return {"objects": [{"id": i, "shape": o["shape"], "colour": o["colour"]}
                            for i, o in enumerate(self._truth.objects)],
                "condition": {"temp": self._cond[0], "light": self._cond[1]},
                "interactions": list(INTERACTIONS),
                "budget_left": self.budget_left}

    def _spend(self):
        if self._used >= self._budget:
            raise BudgetExhausted()
        self._used += 1

    def _check_object(self, i):
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(self._truth.objects):
            raise ValueError("invalid object id: %r" % (i,))

    def step(self, action):
        """Validate and execute one action. Invalid actions raise ValueError and cost nothing."""
        if not isinstance(action, dict) or "type" not in action:
            raise ValueError("action must be a dict with a 'type'")
        kind = action["type"]
        if kind == "inspect":
            i = action.get("object")
            self._check_object(i)
            self._spend()
            o = self._truth.objects[i]
            return {"id": i, "shape": o["shape"], "colour": o["colour"], "material": o["material"]}
        if kind == "set_condition":
            temp, light = action.get("temp"), action.get("light")
            if temp not in TEMPS or light not in LIGHTS:
                raise ValueError("invalid condition")
            self._spend()
            self._cond = (temp, light)
            return {"condition": {"temp": temp, "light": light}}
        if kind == "interact":
            i, inter = action.get("object"), action.get("interaction")
            self._check_object(i)
            if inter not in INTERACTIONS:
                raise ValueError("invalid interaction: %r" % (inter,))
            self._spend()
            feats = dict(self._truth.objects[i])
            feats["temp"], feats["light"] = self._cond
            return {"outcome": self._truth.outcome(inter, feats)}
        raise ValueError("unknown action type: %r" % (kind,))
