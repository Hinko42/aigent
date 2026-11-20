"""Sequential learning experiments with controlled environmental changes."""

import argparse
import json

from .environment import Env, GroundTruth
from .evaluator import evaluate
from .learner import RuleLearner
from .strategies import SystematicStrategy, View
from .world_spec import CLASSIC_WORLD


def collect_phase(env, budget, spec):
    """Collect one reproducible sequence of observations."""

    view = View(env.observe(), spec)
    strategy = SystematicStrategy()
    dummy = RuleLearner(spec)

    records = []
    steps = 0

    while steps < budget and env.budget_left > 0:
        action = strategy.choose(view, dummy, None)
        result = env.step(action)
        steps += 1

        if action["type"] == "inspect":
            object_id = action["object"]

            view.objects[object_id].update({
                key: value
                for key, value in result.items()
                if key in spec.object_features
            })

        elif action["type"] == "set_condition":
            view.cond = tuple(
                action[name]
                for name in spec.condition_features
            )

        else:
            object_id = action["object"]

            records.append({
                "object_id": object_id,
                "interaction": action["interaction"],
                "features": view.features(object_id),
                "outcome": result["outcome"],
            })

    return records


def train(learner, records):
    for record in records:
        learner.update(
            record["interaction"],
            record["features"],
            record["outcome"],
            object_id=record["object_id"],
        )


def snapshot(env):
    """Create an independent ground-truth snapshot for evaluation."""

    truth = env.truth()

    return GroundTruth(
        objects=[dict(obj) for obj in truth.objects],
        rules=list(truth.rules),
        spec=truth.spec,
    )


def run_experiment(
    seed=0,
    phase_budget=60,
    n_changed=1,
    spec=CLASSIC_WORLD,
):
    env = Env(
        seed,
        budget=phase_budget * 2,
        spec=spec,
    )

    # Phase A
    truth_a = snapshot(env)

    records_a = collect_phase(
        env,
        phase_budget,
        spec,
    )

    persistent = RuleLearner(spec)
    train(persistent, records_a)

    # Controlled environmental change
    changed_indices = env.change_rules(
        seed=seed + 100000,
        n_changed=n_changed,
    )

    truth_b = snapshot(env)

    # Phase B: identical observation schedule
    records_b = collect_phase(
        env,
        phase_budget,
        spec,
    )

    # Persistent system retains Phase A evidence.
    train(persistent, records_b)

    # Stateless system starts Phase B from scratch.
    stateless = RuleLearner(spec)
    train(stateless, records_b)

    return {
        "seed": seed,
        "changed_indices": changed_indices,
        "phase_a_observations": len(records_a),
        "phase_b_observations": len(records_b),
        "stateless": evaluate(
            truth_b,
            stateless,
            performed=set(),
        ),
        "persistent": evaluate(
            truth_b,
            persistent,
            performed=set(),
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--changed", type=int, default=1)

    args = parser.parse_args()

    results = [
        run_experiment(
            seed=i,
            phase_budget=args.budget,
            n_changed=args.changed,
        )
        for i in range(args.seeds)
    ]

    metrics = (
        "space_nontrivial_accuracy",
        "space_false_alarm",
        "rules_recovered",
        "exact_recall",
        "n_spurious",
    )

    print("\nSEQUENTIAL LEARNING")
    print(
        f"Seeds: {args.seeds} | "
        f"Phase budget: {args.budget} | "
        f"Changed rules: {args.changed}"
    )

    for system in ("stateless", "persistent"):
        print(f"\n{system.upper()}")

        for metric in metrics:
            values = [
                run[system][metric]
                for run in results
                if run[system][metric] is not None
            ]

            if values:
                print(
                    f"{metric:32s} "
                    f"{sum(values) / len(values):.4f}"
                )

    output = {
        "config": vars(args),
        "results": results,
    }

    with open("sequential_results.json", "w") as file:
        json.dump(output, file, indent=2)

    print("\nSaved sequential_results.json")


if __name__ == "__main__":
    main()