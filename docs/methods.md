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

Version 0.1 evaluates supplied candidates and monitors intervention effectiveness.
A discovery extension would screen candidates or groups, select them on separate
development data, and test removal/control/rescue on fresh states. It should be
compared with exhaustive search where possible, random search, and ordinary
performance ablation at matched search cost. Current research states are
development cases, not unseen validation.

New capabilities require a task adapter (training data and readouts) and a
mechanism adapter (what is changed and how restoration works). The interfaces need
not assume induction, but the current paper does not validate general prerequisite
discovery or a non-induction family. API stability, benchmark versions, and
scientific coverage are separate release properties.

One prospective application is checking whether model compression preserves
current performance while reducing future acquisition. That would require matched
quality, data, adaptation, and compute comparisons; it is not a current finding.
