"""Run from the folder that contains 'exploration':
    py -3 -m unittest exploration.test_exploration -v
"""
import random
import unittest

from .environment import (BudgetExhausted, CONDITIONS, Env, INTERACTIONS, NOTHING,
                          conj_conflict)
from .evaluator import evaluate
from .learner import CANDIDATES, RuleLearner
from .run_stage1 import run_episode
from .strategies import (ActiveCountStrategy, ActiveStrategy, RandomStrategy,
                         SystematicStrategy, View)


class OracleLearner:
    def __init__(self, truth):
        self.truth = truth

    def predict(self, interaction, features):
        return self.truth.outcome(interaction, features)

    def rules(self):
        return list(self.truth.rules)


class ConstantLearner:
    def predict(self, interaction, features):
        return NOTHING

    def rules(self):
        return []


class TestEnvironment(unittest.TestCase):
    def test_same_seed_same_world(self):
        a, b = Env(7), Env(7)
        self.assertEqual(a.truth().rules, b.truth().rules)
        self.assertEqual(a.truth().objects, b.truth().objects)

    def test_five_rule_generation_is_reliable(self):
        """Every benchmark environment must contain all requested hidden rules."""
        for seed in range(100, 130):
            env = Env(
                seed,
                n_objects=12,
                n_rules=5,
                budget=100
            )

            self.assertEqual(
                len(env.truth().rules),
                5,
                f"Seed {seed} generated fewer than five rules"
            )

    def test_different_seeds_differ(self):
        worlds = {tuple(Env(s).truth().rules) for s in range(20)}
        self.assertGreater(len(worlds), 10)

    def test_rules_are_reachable_and_non_overlapping(self):
        for seed in range(60):
            env = Env(seed)
            truth = env.truth()
            self.assertEqual(len(truth.rules), 3)
            qs = truth.queries()
            for r in truth.rules:
                self.assertTrue(any(q["rule"] == r for q in qs), "unreachable rule")
            for i, r1 in enumerate(truth.rules):
                for r2 in truth.rules[i + 1:]:
                    if r1[0] == r2[0]:
                        self.assertTrue(conj_conflict(r1[1], r2[1]))

    def test_observation_hides_material_and_rules(self):
        env = Env(3)
        obs = env.observe()
        for o in obs["objects"]:
            self.assertNotIn("material", o)
        self.assertNotIn("rules", obs)
        self.assertFalse(hasattr(env, "rules"))

    def test_inspect_reveals_material(self):
        env = Env(3)
        res = env.step({"type": "inspect", "object": 0})
        self.assertEqual(res["material"], env.truth().objects[0]["material"])

    def test_budget_is_enforced(self):
        env = Env(1, budget=3)
        for _ in range(3):
            env.step({"type": "inspect", "object": 0})
        self.assertEqual(env.budget_left, 0)
        with self.assertRaises(BudgetExhausted):
            env.step({"type": "inspect", "object": 0})

    def test_invalid_actions_rejected_and_free(self):
        env = Env(1, budget=5)
        bad = [None, {}, {"type": "fly"}, {"type": "inspect", "object": 99},
               {"type": "inspect", "object": True},
               {"type": "interact", "object": 0, "interaction": "explode"},
               {"type": "set_condition", "temp": "lukewarm", "light": "dim"}]
        for action in bad:
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.budget_left, 5)

    def test_interaction_matches_truth(self):
        env = Env(5)
        truth = env.truth()
        temp_light = (env.observe()["condition"]["temp"], env.observe()["condition"]["light"])
        for i in range(len(truth.objects)):
            for inter in INTERACTIONS:
                feats = dict(truth.objects[i])
                feats["temp"], feats["light"] = temp_light
                env_i = Env(5, budget=100)
                res = env_i.step({"type": "interact", "object": i, "interaction": inter})
                self.assertEqual(res["outcome"], truth.outcome(inter, feats))


class TestLearner(unittest.TestCase):
    def test_true_rules_are_never_refuted(self):
        """Soundness: whatever is observed, the true rules stay among the live candidates."""
        for seed in range(25):
            env = Env(seed, budget=300)
            truth = env.truth()
            learner = RuleLearner()
            view = View(env.observe())
            rng = random.Random(seed)
            strat = RandomStrategy()
            while env.budget_left > 0:
                act = strat.choose(view, learner, rng)
                res = env.step(act)
                if act["type"] == "inspect":
                    view.objects[act["object"]]["material"] = res["material"]
                elif act["type"] == "set_condition":
                    view.cond = (act["temp"], act["light"])
                else:
                    learner.update(act["interaction"], view.features(act["object"]),
                                   res["outcome"])
            for r in truth.rules:
                self.assertTrue(learner.alive[CANDIDATES.index(r)],
                                "true rule refuted (seed %d)" % seed)

    def test_unknown_feature_does_not_refute(self):
        learner = RuleLearner()
        feats = {"shape": "cube", "colour": "red", "material": None,
                 "temp": "hot", "light": "dim"}
        learner.update("heat", feats, NOTHING)
        for idx, (inter, conj, eff) in enumerate(CANDIDATES):
            if inter == "heat" and any(f == "material" for f, _ in conj):
                self.assertTrue(learner.alive[idx])

    def test_learns_single_rule_from_data(self):
        learner = RuleLearner()
        wood = {"shape": "cube", "colour": "red", "material": "wood", "temp": "hot", "light": "dim"}
        metal = dict(wood, material="metal")
        learner.update("heat", wood, "melts")
        learner.update("heat", metal, NOTHING)
        self.assertEqual(learner.predict("heat", wood), "melts")
        self.assertEqual(learner.predict("heat", metal), NOTHING)

    def test_late_inspection_revises_old_observations(self):
        learner = RuleLearner()

        unknown = {
            "shape": "cube",
            "colour": "red",
            "material": None,
            "temp": "hot",
            "light": "dim",
        }

        wood_rule = (
            "heat",
            (("material", "wood"),),
            "melts"
        )

        metal_rule = (
            "heat",
            (("material", "metal"),),
            "melts"
        )

        wood_idx = CANDIDATES.index(wood_rule)
        metal_idx = CANDIDATES.index(metal_rule)

        # Record an observation without knowing the material.
        learner.update(
            "heat",
            unknown,
            NOTHING,
            object_id=0
        )

        # Both hypotheses remain possible while material is unknown.
        self.assertTrue(learner.alive[wood_idx])
        self.assertTrue(learner.alive[metal_idx])

        # Later, discover that the object is made of wood.
        learner.update_object_features(
            0,
            {"material": "wood"}
        )

        # The old observation must now be reconsidered.
        self.assertFalse(learner.alive[wood_idx])

        # The metal hypothesis remains possible.
        self.assertTrue(learner.alive[metal_idx])

        # Historical observation was retained.
        self.assertEqual(learner.n_obs, 1)


class TestStrategies(unittest.TestCase):
    def test_strategies_only_emit_valid_actions_within_budget(self):
        for cls in (RandomStrategy, SystematicStrategy, ActiveCountStrategy, ActiveStrategy):
            for seed in range(8):
                out = run_episode(cls, seed, budget=60)
                self.assertIn("rules_recovered", out["final"])

    def test_active_never_repeats_an_experiment(self):
        env = Env(2, budget=60)
        learner = RuleLearner()
        view = View(env.observe())
        rng = random.Random(0)
        strat = ActiveStrategy()
        seen = set()
        while env.budget_left > 0:
            act = strat.choose(view, learner, rng)
            res = env.step(act)
            if act["type"] == "inspect":
                view.objects[act["object"]]["material"] = res["material"]
            elif act["type"] == "set_condition":
                view.cond = (act["temp"], act["light"])
            else:
                key = (act["object"], act["interaction"], view.cond)
                self.assertNotIn(key, seen)
                seen.add(key)
                view.tried.add(key)
                learner.update(act["interaction"], view.features(act["object"]), res["outcome"])


class TestEvaluator(unittest.TestCase):
    def test_oracle_scores_perfectly(self):
        truth = Env(4).truth()
        res = evaluate(truth, OracleLearner(truth), performed=set())
        self.assertEqual(res["space_accuracy"], 1.0)
        self.assertEqual(res["unseen_accuracy"], 1.0)
        self.assertEqual(res["rules_recovered"], 1.0)
        self.assertEqual(res["exact_recall"], 1.0)
        self.assertEqual(res["n_spurious"], 0)
        self.assertEqual(res["space_false_alarm"], 0.0)

    def test_know_nothing_learner(self):
        truth = Env(4).truth()
        res = evaluate(truth, ConstantLearner(), performed=set())
        self.assertEqual(res["rules_recovered"], 0.0)
        self.assertEqual(res["space_nontrivial_accuracy"], 0.0)
        self.assertEqual(res["space_false_alarm"], 0.0)

    def test_performed_experiments_are_excluded_from_unseen(self):
        truth = Env(4).truth()
        performed = {q["id"] for q in truth.queries()[:10]}
        res = evaluate(truth, OracleLearner(truth), performed)
        self.assertEqual(res["n_unseen"], len(truth.queries()) - 10)


if __name__ == "__main__":
    unittest.main()
