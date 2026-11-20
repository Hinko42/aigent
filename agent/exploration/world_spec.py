"""Public specifications describing the structure of an exploration world."""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class WorldSpec:
    name: str
    features: dict
    object_features: tuple
    observable_features: tuple
    condition_features: tuple
    interactions: tuple
    effects: tuple
    max_rule_arity: int = 2

    @property
    def feature_order(self):
        return tuple(self.features.keys())

    @property
    def conditions(self):
        values = [
            self.features[name]
            for name in self.condition_features
        ]
        return tuple(product(*values))


CLASSIC_WORLD = WorldSpec(
    name="classic",

    features={
        "shape": ("cube", "sphere", "cone"),
        "material": ("wood", "metal", "glass"),
        "colour": ("red", "blue", "green"),
        "temp": ("cold", "hot"),
        "light": ("dim", "bright"),
    },

    object_features=("shape", "material", "colour"),
    observable_features=("shape", "colour"),
    condition_features=("temp", "light"),

    interactions=("heat", "strike", "soak", "shine"),
    effects=("melts", "breaks", "glows", "fizzes", "darkens"),

    max_rule_arity=2,
)


STRANGE_WORLD = WorldSpec(
    name="strange",

    features={
        "species": ("moss", "crystal", "fungus", "lichen"),
        "density": ("porous", "dense", "fluid"),
        "origin": ("cavern", "surface", "thermal"),
        "temp": ("cold", "hot"),
        "light": ("dim", "bright"),
    },

    object_features=("species", "density", "origin"),
    observable_features=("species", "origin"),
    condition_features=("temp", "light"),

    interactions=("heat", "strike", "soak", "shine"),
    effects=("melts", "breaks", "glows", "fizzes", "darkens"),

    max_rule_arity=2,
)


THIRD_WORLD = WorldSpec(
    name="third",

    features={
        "lattice": ("woven", "layered", "porous"),
        "substrate": ("ceramic", "polymer", "alloy"),
        "provenance": ("abyssal", "terrestrial", "extrusive"),
        "pressure": ("low", "balanced", "extreme"),
        "moisture": ("arid", "saturated"),
    },

    object_features=("lattice", "substrate", "provenance"),
    observable_features=("lattice", "provenance"),
    condition_features=("pressure", "moisture"),

    interactions=("compress", "vibrate", "irradiate"),
    effects=("fractures", "swells", "conducts", "crystallizes"),

    max_rule_arity=3,
)