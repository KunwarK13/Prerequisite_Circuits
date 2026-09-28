# Prerequisite Circuits

[![Tests](https://github.com/KunwarK13/Prerequisite_Circuits/actions/workflows/test.yml/badge.svg)](https://github.com/KunwarK13/Prerequisite_Circuits/actions/workflows/test.yml)

**Track what an intervention removes—and whether it stays removed as a model learns.**

Code and evidence accompanying *When Data Can Teach: Prerequisite Circuits Gate
Learning*, by **Kunwar Kalra and Thanush Patlolla**. Submitted to ICLR 2027;
an arXiv link will be added when available.

[Paper](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/paper/main.pdf) · [Reproduction](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/docs/reproduction.md) ·
[Package guide](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/docs/package.md) · [Evidence](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/docs/evidence.md)

The research studies how an existing computation changes what fixed future data
can teach. Controlled removal/rescue experiments isolate this effect. Pretrained
Pythia experiments test training-only head suppression and examine compensation
over longer training. The demonstrated task family is induction and copying.

![Controlled training-only suppression and activation rescue](https://raw.githubusercontent.com/KunwarK13/Prerequisite_Circuits/main/docs/assets/removal-rescue.png)

## Start with the evidence

The package's reporting API uses only the Python standard library. From a clone:

```bash
python -m pip install .
prereq report results/reference/410m_*.json --html outputs/report.html --open
```

The interactive report works offline: compare runs, inspect exact observations,
and export the embedded records. It reads saved evidence and does not train a model. For the paper's
figures, raw-record verification, and experiment recipes, see
[Reproduction](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/docs/reproduction.md).

## Run a small experiment

```bash
python -m pip install '.[train,plot]'
OMP_NUM_THREADS=1 prereq demo --output outputs/demo
prereq report outputs/demo/*.json --plot outputs/demo/learning.pdf
```

The CPU example compares available, suppressed, matched-control, correct-supply,
and wrong-supply conditions under common evaluation. It is an illustrative
fixed-feature model, **not a replication or additional result from the paper**.
The paper's controlled-transformer replication is available separately.

## Use the monitor in your experiment

```python
from prerequisite_circuits import Criterion, Observation, Trajectory

trajectory = Trajectory(
    "head-suppression",
    criterion=Criterion(maximum_suppressed=0.20, minimum_drop=0.30, acquisition=0.50),
    metadata={"evaluation": "training intervention removed"},
)
trajectory.append(Observation(
    step=0,
    target_available=0.01,
    prerequisite_available=0.66,
    prerequisite_suppressed=0.06,
))
trajectory.save("outputs/trajectory.json")
```

These are illustrative values. Supply measurements from your own independent
probe and choose the criterion before evaluating outcomes. The guide shows how to
attach the monitor to a PyTorch loop without replacing its optimizer or trainer.

The optional PyTorch API provides scoped channel/head interventions, detached
activation supply, exact batch replay checks, and a small matched-branch runner.
Hooks are removed on exit, including exceptions. Evaluation preserves random
states, module modes, and buffers. The runner initializes a fresh optimizer in
each branch; it does not silently inherit optimizer history.

## What version 0.1 establishes

The package reports **intervention effectiveness on a declared probe at observed
updates**. It also measures target acquisition under the specified evaluation
condition. It does not infer continuous suppression, discover a complete circuit,
or turn temporal ordering into a causal mechanism.

A candidate can be evaluated with removal, controls, and rescue. Automatic
candidate discovery and validated mechanisms beyond induction are future research
directions. New task/mechanism adapters can be added without changing the record
format. See [Methods and scope](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/docs/methods.md).

## Repository

| Path | Contents |
|---|---|
| `src/prerequisite_circuits/` | Reusable monitoring, interventions, replay, and reports |
| `experiments/` | Versioned paper recipes and recorded environments |
| `analysis/` | Figure/table reproduction and evidence checks |
| `results/` | Compact reference trajectories and artifact manifests |
| `examples/` | Small working examples |
| `tests/` | Software and intervention-semantics tests |
| `paper/` | Preprint PDF, complete LaTeX source, figures, and generated tables |
| `docs/` | Reproduction, package usage, methods, and evidence index |

Original execution sources and larger records are separate, checksum-verified
release assets. They retain their historical identities. Refactored package code
is a new implementation with explicit parity tests, not a relabeled historical
execution. No training or multi-GB download occurs on import or installation.

## Development

See [Development and releases](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/docs/development.md) for the public/private research
workflow and publication steps.

```bash
python -m pip install -e '.[dev,pythia]'
python -m pytest
ruff check src tests examples analysis experiments
python -m build
```

The scientific checks include channel/gradient scope, restoration cleanup,
data-replay mismatches, observation neutrality, and independent reconstruction of
GPT-NeoX head outputs. GPU experiments are opt-in. Original code is MIT licensed;
third-party assets retain their own licenses. See [CITATION.cff](https://github.com/KunwarK13/Prerequisite_Circuits/blob/main/CITATION.cff).
