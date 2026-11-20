# Exploration, stage 1: environment, baselines, evaluator (no LLM yet)

Standard library only. Nothing here touches the existing experiments (`workspace/`, `benchmark/`,
`orchestrator.py`, `overnight_queue.py`).

## What this stage is for
It defines what "learning the hidden rules" means and measures how well simple explorers do it,
so that a later LLM agent has something concrete to be compared against.

## Files
- `environment.py`: procedurally generated world. 8 objects (shape, material, colour), world
  conditions (temp, light), 4 interactions, 5 possible effects, and 3 hidden rules per world.
  A rule is `(interaction, conjunction of 1-2 feature=value pairs, effect)`, e.g.
  `("heat", (("material","wood"), ("temp","hot")), "melts")`. Rules of one interaction never
  overlap, and every rule can be triggered by at least one object.
- `learner.py`: inference shared by all explorers. Keeps every candidate rule still consistent
  with the observations (1,600 candidates), then explains the observed effects with as few simple
  rules as possible. It never sees the hidden rules.
- `strategies.py`: `random`, `systematic` (inspect all, then sweep condition by condition),
  `active_count` and `active` (greedy; the latter by expected information gain).
- `evaluator.py`: reads the ground truth. The explorers never do.
- `run_stage1.py`: runs every strategy on the same environments and prints the comparison.
- `test_exploration.py`: 16 unit tests.

## Action format (also the planned interface for the LLM agent)
```
{"type": "inspect", "object": 3}                              cost 1, reveals the material
{"type": "set_condition", "temp": "hot", "light": "dim"}      cost 1
{"type": "interact", "object": 3, "interaction": "heat"}      cost 1, returns {"outcome": ...}
```
`Env.observe()` is free and shows shape and colour only. Invalid actions raise `ValueError` and
cost nothing. Every action beyond the budget raises `BudgetExhausted`.

## Commands (Windows, from the folder that contains `exploration`, e.g. C:\agent)
```
py -3 -m unittest exploration.test_exploration -v
py -3 -m exploration.run_stage1 --seeds 30
py -3 -m exploration.run_stage1 --seeds 30 --first-seed 100
```
Options: `--budget 60 --rules 3 --objects 8`. Results are saved as JSON in `exploration\results\`.

## Metrics
- `space_accuracy`: accuracy over all 432 combinations of object features, conditions and
  interactions. The same test set for every explorer, so it is the fair comparison.
- `space_nontrivial_accuracy`, `space_false_alarm`: the same, split into cases where something
  happens and cases where nothing does.
- `rules_recovered`: fraction of hidden rules whose every firing case is predicted correctly.
- `exact_recall`, `n_spurious`: exact match of the learned rules against the hidden ones.
- `unseen_accuracy` is computed but not used for comparing strategies: each strategy leaves a
  different set of experiments unperformed, so the test sets differ.

## Limits
- The hidden rules live in the same Python process. A strategy written by us cannot reach them
  by accident, but this is not a security boundary. Any LLM agent must only ever receive the
  JSON interface above.
- The learner and the active strategies were changed after looking at seeds 0-29. Seeds 100-129
  were run afterwards as a check and agree on the main result, but they are a check on this
  small world, not evidence about anything bigger.
- Not built yet: calibration, transfer between worlds, persistent memory, the LLM agent and the
  curiosity signals. Those are stages 2-4.
- Nothing here measures anything like awareness. It measures how fast rules are found.
