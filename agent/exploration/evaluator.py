"""Independent evaluator. It reads the ground truth; explorers never do.

Strategy-independent metrics (same test set for every explorer):
  space_accuracy             accuracy over the whole attribute space (432 experiments)
  space_nontrivial_accuracy  same, only where something actually happens
  space_false_alarm          fraction of 'nothing' cases where it predicted an effect
  rules_recovered            fraction of hidden rules whose every firing case is predicted right
  exact_recall / n_spurious  exact match of learned rules against the hidden ones

Strategy-dependent metric (reported, but not for comparing strategies, because each strategy
leaves a different set of experiments unperformed):
  unseen_accuracy            accuracy on reachable experiments the explorer never performed
"""
from .environment import NOTHING


def evaluate(truth, learner, performed):
    qs = truth.queries()
    space = truth.space_queries()

    def pred(q):
        return learner.predict(q["interaction"], q["features"])

    def acc(sub):
        if not sub:
            return None
        return sum(1 for q in sub if pred(q) == q["truth"]) / len(sub)

    unseen = [q for q in qs if q["id"] not in performed]
    s_nontrivial = [q for q in space if q["truth"] != NOTHING]
    s_trivial = [q for q in space if q["truth"] == NOTHING]

    recovered = 0
    for r in truth.rules:
        firing = [q for q in qs if q["rule"] == r]
        if firing and all(pred(q) == r[2] for q in firing):
            recovered += 1

    learned = learner.rules()
    true_rules = set(truth.rules)
    return {
        "space_accuracy": acc(space),
        "space_nontrivial_accuracy": acc(s_nontrivial),
        "space_false_alarm": (None if not s_trivial else
                              sum(1 for q in s_trivial if pred(q) != NOTHING) / len(s_trivial)),
        "rules_recovered": recovered / len(truth.rules) if truth.rules else None,
        "exact_recall": (len(true_rules & set(learned)) / len(true_rules)) if true_rules else None,
        "n_spurious": sum(1 for r in learned if r not in true_rules),
        "unseen_accuracy": acc(unseen),
        "n_unseen": len(unseen),
    }
