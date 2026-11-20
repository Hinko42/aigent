"""Tests for world-independent exploration and learning."""

import unittest

from .world_spec import (
    CLASSIC_WORLD,
    STRANGE_WORLD,
    THIRD_WORLD,
)
from .environment import Env
from .learner import RuleLearner
from .strategies import (
    RandomStrategy,
    SystematicStrategy,
    ActiveCountStrategy,
    ActiveStrategy,
)
from .run_stage1 import run_episode
from .evaluator import evaluate


STRATEGIES = (
    RandomStrategy,
    SystematicStrategy,
    ActiveCountStrategy,
    ActiveStrategy,
)


class OracleLearner:
    def __init__(self, truth):
        self.truth = truth

    def predict(self, interaction, features):
        return self.truth.outcome(interaction, features)

    def rules(self):
        return list(self.truth.rules)


class TestWorldGeneralization(unittest.TestCase):

    def test_both_worlds_generate_requested_rules(self):
        for spec in (CLASSIC_WORLD, STRANGE_WORLD, THIRD_WORLD):
            for seed in range(100, 130):
                env = Env(
                    seed,
                    n_objects=12,
                    n_rules=5,
                    budget=100,
                    spec=spec,
                )

                self.assertEqual(
                    len(env.truth().rules),
                    5,
                    f"{spec.name}, seed {seed}",
                )

    def test_strange_world_uses_its_own_features(self):
        env = Env(
            42,
            n_objects=8,
            n_rules=3,
            spec=STRANGE_WORLD,
        )

        truth = env.truth()

        for obj in truth.objects:
            self.assertEqual(
                set(obj.keys()),
                set(STRANGE_WORLD.object_features),
            )

        for rule in truth.rules:
            for feature, value in rule[1]:
                self.assertIn(
                    feature,
                    STRANGE_WORLD.features,
                )

                self.assertIn(
                    value,
                    STRANGE_WORLD.features[feature],
                )

    def test_hidden_features_are_not_observed(self):
        env = Env(
            42,
            spec=STRANGE_WORLD,
        )

        obs = env.observe()

        for obj in obs["objects"]:
            self.assertNotIn("density", obj)
            self.assertIn("species", obj)
            self.assertIn("origin", obj)

        result = env.step({
            "type": "inspect",
            "object": 0,
        })

        self.assertIn("density", result)

    def test_learner_builds_world_specific_candidates(self):
        learner = RuleLearner(STRANGE_WORLD)

        self.assertGreater(len(learner.candidates), 0)

        for interaction, conjunction, effect in learner.candidates:
            self.assertIn(
                interaction,
                STRANGE_WORLD.interactions,
            )

            self.assertIn(
                effect,
                STRANGE_WORLD.effects,
            )

            for feature, value in conjunction:
                self.assertIn(
                    feature,
                    STRANGE_WORLD.features,
                )

                self.assertIn(
                    value,
                    STRANGE_WORLD.features[feature],
                )

        candidate_features = {
            feature
            for _, conjunction, _ in learner.candidates
            for feature, _ in conjunction
        }

        self.assertIn("species", candidate_features)
        self.assertIn("density", candidate_features)
        self.assertIn("origin", candidate_features)

        self.assertNotIn("shape", candidate_features)
        self.assertNotIn("material", candidate_features)

    def test_strange_world_has_no_duplicate_space_queries(self):
        env = Env(
            42,
            spec=STRANGE_WORLD,
        )

        queries = env.truth().space_queries()

        expected = (
            4 * 3 * 3
            * 4
            * len(STRANGE_WORLD.interactions)
        )

        self.assertEqual(len(queries), expected)

        identifiers = {
            (
                q["interaction"],
                tuple(sorted(q["features"].items())),
            )
            for q in queries
        }

        self.assertEqual(len(identifiers), expected)

    def test_all_strategies_complete_strange_world_episodes(self):
        for strategy in STRATEGIES:
            for seed in range(5):
                result = run_episode(
                    strategy,
                    seed,
                    budget=60,
                    n_rules=3,
                    n_objects=8,
                    spec=STRANGE_WORLD,
                )

                self.assertIn(
                    "rules_recovered",
                    result["final"],
                )

                self.assertIn(
                    "space_accuracy",
                    result["final"],
                )

    def test_oracle_evaluation_works_in_both_worlds(self):
        for spec in (CLASSIC_WORLD, STRANGE_WORLD, THIRD_WORLD):
            truth = Env(
                42,
                spec=spec,
            ).truth()

            result = evaluate(
                truth,
                OracleLearner(truth),
                performed=set(),
            )

            self.assertEqual(
                result["space_accuracy"],
                1.0,
            )

            self.assertEqual(
                result["rules_recovered"],
                1.0,
            )

            self.assertEqual(
                result["exact_recall"],
                1.0,
            )

            self.assertEqual(
                result["n_spurious"],
                0,
            )
    
    def test_third_world_uses_three_feature_rules(self):
        env = Env(
            42,
            n_objects=12,
            n_rules=5,
            spec=THIRD_WORLD,
        )

        truth = env.truth()

        for obj in truth.objects:
            self.assertEqual(
                set(obj.keys()),
                set(THIRD_WORLD.object_features),
            )

        for rule in truth.rules:
            self.assertLessEqual(len(rule[1]), 3)

            for feature, value in rule[1]:
                self.assertIn(feature, THIRD_WORLD.features)
                self.assertIn(
                    value,
                    THIRD_WORLD.features[feature],
                )

        learner = RuleLearner(THIRD_WORLD)

        self.assertTrue(
            any(
                len(conjunction) == 3
                for _, conjunction, _ in learner.candidates
            )
        )

    def test_all_strategies_complete_third_world_episodes(self):
        for strategy in STRATEGIES:
            for seed in range(5):
                result = run_episode(
                    strategy,
                    seed,
                    budget=60,
                    n_rules=3,
                    n_objects=8,
                    spec=THIRD_WORLD,
                )

                self.assertIn(
                    "rules_recovered",
                    result["final"],
                )

                self.assertIn(
                    "space_accuracy",
                    result["final"],
                )

if __name__ == "__main__":
    unittest.main()