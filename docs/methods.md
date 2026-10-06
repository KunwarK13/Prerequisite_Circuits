# Interpretation and extension

The prerequisite is a computation. A head or group of heads is a candidate
implementation. Suppressing a selected component initially need not prevent other
components from implementing a related computation later.

## What to measure

1. Fix a starting checkpoint, ordered target data, training policy, observation
   grid, independent readouts, and the criterion for meaningful acquisition.
2. Verify that the available and matched-control arms learn within the budget.
3. Select candidate interventions without using the final target outcomes.
4. Measure suppression at the start and throughout training, alongside target
   acquisition and background-task controls.
5. Evaluate without training-only interventions. For stronger restoration claims,
   verify that the restored computation and its input interface remain matched.
6. When feasible, compare correct activation supply against absent and wrong
   supply under the same evaluation policy.

The package automates measurements and checks selected execution invariants. The
researcher remains responsible for probe validity, selective interventions,
controls, independent replications, and the causal interpretation.

## Two evaluation settings

**Fixed computation:** the relevant parameters and their input dependencies are
held fixed across arms; restoration is numerically checked. Shared restrictions
must still allow the positive controls to learn.

**Changing trajectory:** the mask is removed at evaluation, but the model and its
upstream representations have changed during training. This can establish an
effect of the intervention over training without restoring an unchanged
computation. The pretrained Pythia setting requires this distinction.

Neither low probe accuracy nor the order of recovery and acquisition proves that
all alternative implementations were removed. A failed positive control makes a
prerequisite interpretation inconclusive. A finite non-acquisition horizon is not
proof that learning is impossible.

## Future releases

Keep one package and extend it as the evidence improves. These are milestones,
not promised release dates:

| Stage | Addition | Release requirement |
|---|---|---|
| 0.1, available | Audit supplied candidates, apply scoped interventions, check replay, and inspect learning trajectories. | Tested software semantics and an explicit account of what each probe measures. |
| Next maintenance releases | Fix bugs and improve integration with existing trainers. | Regression tests; preserve old reports or document a migration. |
| Discovery release, if validated | Rank candidate components and groups for causal follow-up. | Predict held-out intervention outcomes at useful search cost, then confirm removal/control/rescue on fresh states. |
| Broader-mechanism release, if validated | Add a distinct task and mechanism, such as a candidate MLP feature supporting factual acquisition. | Intact learning, selective intervention, appropriate restoration and rescue, and independent replications. |
| 1.0 | Commit to a stable API and report schema. | Compatibility and maintenance guarantees; this is not a claim of universal scientific coverage. |

The central research problem is selective removal: disable the proposed
computation while preserving the model's ability to learn when that computation
is available. Redundant implementations and recovery during training mean that
removing a fixed set of heads may cease to remove the computation. More extensive
damage can block learning for unrelated reasons. Independent probes, matched
controls, and activation rescue help distinguish these cases.

Start discovery in the controlled model, where exhaustive intervention search is
feasible. Test groups as well as single heads; compare short-branch screening with
random search and ordinary performance ablation at matched total search cost.
Freeze selection rules before evaluating fresh states. Current research states
are development cases, not unseen validation. A useful candidate ranking does
not establish that a complete or unique circuit has been identified.

New capabilities need a task adapter (training data and readouts) and a mechanism
adapter (what changes and how restoration works). First establish that the intact
model learns on held-out examples within a fixed budget. Then validate removal
and rescue without selecting interventions on final outcomes. Probes must detect
the intended computation rather than a task shortcut; supplied activations must
remain compatible with the receiving model. Supporting another architecture alone
does not establish a new prerequisite family.

Task benchmarks, report schemas, and package versions remain separately versioned.
Old evidence and releases stay reproducible when a new adapter is added.

One prospective application is checking whether model compression preserves
current performance while reducing future acquisition. That would require matched
quality, data, adaptation, and compute comparisons; it is not a current finding.
