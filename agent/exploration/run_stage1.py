"""Run all explorers on the same environments and compare them.

From the folder that contains 'exploration':
    py -3 -m exploration.run_stage1 --seeds 30
"""
import argparse
import json
import os
import random

from .environment import Env
from .evaluator import evaluate
from .learner import RuleLearner
from .strategies import (ActiveCountStrategy, ActiveStrategy, RandomStrategy,
                         SystematicStrategy, View)

STRATEGIES = (RandomStrategy, SystematicStrategy, ActiveCountStrategy, ActiveStrategy)
METRICS = ("rules_recovered", "exact_recall", "space_accuracy",
           "space_nontrivial_accuracy", "space_false_alarm", "n_spurious")


def run_episode(strategy_cls, seed, budget=60, n_rules=3, n_objects=8, checkpoints=()):
    env = Env(seed, n_objects=n_objects, n_rules=n_rules, budget=budget)
    truth = env.truth()
    learner = RuleLearner()
    view = View(env.observe())
    rng = random.Random(100000 + seed)
    strategy = strategy_cls()
    performed = set()
    curve = {}
    steps = 0
    while env.budget_left > 0:
        act = strategy.choose(view, learner, rng)
        res = env.step(act)
        steps += 1
        if act["type"] == "inspect":
            view.objects[act["object"]]["material"] = res["material"]
        elif act["type"] == "set_condition":
            view.cond = (act["temp"], act["light"])
        else:
            i, inter = act["object"], act["interaction"]
            learner.update(inter, view.features(i), res["outcome"])
            performed.add((i, inter) + view.cond)
            view.tried.add((i, inter, view.cond))
        if steps in checkpoints:
            curve[steps] = evaluate(truth, learner, performed)
    return {"final": evaluate(truth, learner, performed), "curve": curve,
            "n_performed": len(performed)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fmt(x):
    return "-" if x is None else "%.3f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--rules", type=int, default=3)
    ap.add_argument("--objects", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results", "stage1_results.json"))
    args = ap.parse_args()
    checkpoints = tuple(c for c in (10, 20, 30, 40, 50, 60, 80, 100) if c <= args.budget)

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    results = {}
    for cls in STRATEGIES:
        results[cls.name] = [run_episode(cls, s, args.budget, args.rules, args.objects,
                                         checkpoints) for s in seeds]

    print("Budget %d, %d rules, %d objects, seeds %d-%d (same environments for every strategy)"
          % (args.budget, args.rules, args.objects, seeds[0], seeds[-1]))
    print("\nFINAL (mean +/- sd over seeds)")
    print("%-12s" % "strategy" + "".join("%27s" % m for m in METRICS))
    for name, runs in results.items():
        cells = []
        for m in METRICS:
            vals = [r["final"][m] for r in runs]
            cells.append("%27s" % ("%s +/- %s" % (fmt(mean(vals)), fmt(sd(vals)))))
        print("%-12s" % name + "".join(cells))

    print("\nLEARNING CURVE: mean rules_recovered / space_accuracy after N actions")
    print("%-12s" % "strategy" + "".join("%14d" % c for c in checkpoints))
    for name, runs in results.items():
        cells = []
        for c in checkpoints:
            cells.append("%14s" % ("%s/%s" % (
                fmt(mean([r["curve"][c]["rules_recovered"] for r in runs])),
                fmt(mean([r["curve"][c]["space_accuracy"] for r in runs])))))
        print("%-12s" % name + "".join(cells))

    print("\nPAIRED DIFFERENCES (first minus second; wins/ties/losses for the first)")
    for a, b in (("active", "random"), ("active", "systematic"), ("active", "active_count"),
                 ("active_count", "random"), ("active_count", "systematic"),
                 ("systematic", "random")):
        for m in ("rules_recovered", "space_accuracy"):
            d = [results[a][k]["final"][m] - results[b][k]["final"][m]
                 for k in range(len(seeds))]
            wins = sum(1 for x in d if x > 1e-9)
            losses = sum(1 for x in d if x < -1e-9)
            print("%-12s vs %-12s %-16s mean %+.3f  %d/%d/%d"
                  % (a, b, m, sum(d) / len(d), wins, len(d) - wins - losses, losses))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
