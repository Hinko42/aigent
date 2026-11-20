"""Run explorers against procedurally generated worlds."""

import argparse
import json
import os
import random

from .environment import Env
from .evaluator import evaluate
from .learner import RuleLearner
from .strategies import (
    ActiveCountStrategy,
    ActiveStrategy,
    RandomStrategy,
    SystematicStrategy,
    View,
)
from .world_spec import (
    CLASSIC_WORLD,
    STRANGE_WORLD,
    THIRD_WORLD,
)


STRATEGIES = (
    RandomStrategy,
    SystematicStrategy,
    ActiveCountStrategy,
    ActiveStrategy,
)

METRICS = (
    "rules_recovered",
    "exact_recall",
    "space_accuracy",
    "space_nontrivial_accuracy",
    "space_false_alarm",
    "n_spurious",
)


WORLDS = {
    "classic": CLASSIC_WORLD,
    "strange": STRANGE_WORLD,
    "third": THIRD_WORLD,
}


def run_episode(
    strategy_cls,
    seed,
    budget=60,
    n_rules=3,
    n_objects=8,
    checkpoints=(),
    spec=CLASSIC_WORLD,
):
    env = Env(
        seed,
        n_objects=n_objects,
        n_rules=n_rules,
        budget=budget,
        spec=spec,
    )

    truth = env.truth()
    learner = RuleLearner(spec)
    view = View(env.observe(), spec)

    rng = random.Random(100000 + seed)
    strategy = strategy_cls()

    performed = set()
    curve = {}
    steps = 0

    while env.budget_left > 0:
        action = strategy.choose(view, learner, rng)
        result = env.step(action)

        steps += 1

        if action["type"] == "inspect":
            object_id = action["object"]

            view.objects[object_id].update(result)

            learner.update_object_features(
                object_id,
                view.objects[object_id],
            )

        elif action["type"] == "set_condition":
            view.cond = tuple(
                action[name]
                for name in spec.condition_features
            )

        else:
            object_id = action["object"]
            interaction = action["interaction"]

            learner.update(
                interaction,
                view.features(object_id),
                result["outcome"],
                object_id=object_id,
            )

            performed.add(
                (object_id, interaction) + view.cond
            )

            view.tried.add(
                (object_id, interaction, view.cond)
            )

        if steps in checkpoints:
            curve[steps] = evaluate(
                truth,
                learner,
                performed,
            )

    return {
        "final": evaluate(truth, learner, performed),
        "curve": curve,
        "n_performed": len(performed),
    }


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def sd(values):
    values = [x for x in values if x is not None]

    if len(values) < 2:
        return 0.0

    m = sum(values) / len(values)

    return (
        sum((x - m) ** 2 for x in values)
        / (len(values) - 1)
    ) ** 0.5


def fmt(value):
    return "-" if value is None else "%.3f" % value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--rules", type=int, default=3)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument(
        "--world",
        choices=WORLDS.keys(),
        default="classic",
    )

    parser.add_argument(
        "--out",
        default=None,
    )

    args = parser.parse_args()

    spec = WORLDS[args.world]

    if args.out is None:
        args.out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            f"{args.world}_results.json",
        )

    checkpoints = tuple(
        checkpoint
        for checkpoint in (10, 20, 30, 40, 50, 60, 80, 100)
        if checkpoint <= args.budget
    )

    seeds = list(
        range(
            args.first_seed,
            args.first_seed + args.seeds,
        )
    )

    results = {}

    for strategy_cls in STRATEGIES:
        results[strategy_cls.name] = [
            run_episode(
                strategy_cls,
                seed,
                args.budget,
                args.rules,
                args.objects,
                checkpoints,
                spec,
            )
            for seed in seeds
        ]

    print(
        "\nWorld: %s | Budget: %d | Rules: %d | Objects: %d"
        % (
            spec.name,
            args.budget,
            args.rules,
            args.objects,
        )
    )

    print(
        "Seeds: %d-%d (identical environments for every strategy)"
        % (seeds[0], seeds[-1])
    )

    print("\nFINAL (mean +/- sd over seeds)")
    print(
        "%-12s" % "strategy"
        + "".join("%27s" % metric for metric in METRICS)
    )

    for name, runs in results.items():
        cells = []

        for metric in METRICS:
            values = [
                run["final"][metric]
                for run in runs
            ]

            cells.append(
                "%27s" % (
                    "%s +/- %s"
                    % (fmt(mean(values)), fmt(sd(values)))
                )
            )

        print("%-12s" % name + "".join(cells))

    print("\nLEARNING CURVE: rules_recovered / space_accuracy")

    print(
        "%-12s" % "strategy"
        + "".join("%14d" % checkpoint for checkpoint in checkpoints)
    )

    for name, runs in results.items():
        cells = []

        for checkpoint in checkpoints:
            cells.append(
                "%14s" % (
                    "%s/%s"
                    % (
                        fmt(mean([
                            run["curve"][checkpoint]["rules_recovered"]
                            for run in runs
                        ])),
                        fmt(mean([
                            run["curve"][checkpoint]["space_accuracy"]
                            for run in runs
                        ])),
                    )
                )
            )

        print("%-12s" % name + "".join(cells))

    print("\nPAIRED DIFFERENCES")

    comparisons = (
        ("active", "random"),
        ("active", "systematic"),
        ("active", "active_count"),
        ("active_count", "random"),
        ("active_count", "systematic"),
        ("systematic", "random"),
    )

    for first, second in comparisons:
        for metric in ("rules_recovered", "space_accuracy"):
            differences = [
                results[first][i]["final"][metric]
                - results[second][i]["final"][metric]
                for i in range(len(seeds))
            ]

            wins = sum(value > 1e-9 for value in differences)
            losses = sum(value < -1e-9 for value in differences)
            ties = len(differences) - wins - losses

            print(
                "%-12s vs %-12s %-16s mean %+.3f  %d/%d/%d"
                % (
                    first,
                    second,
                    metric,
                    sum(differences) / len(differences),
                    wins,
                    ties,
                    losses,
                )
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as file:
        json.dump(
            {
                "args": vars(args),
                "world": spec.name,
                "results": results,
            },
            file,
        )

    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()