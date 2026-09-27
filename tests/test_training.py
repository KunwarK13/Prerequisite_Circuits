import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from torch import nn

from prerequisite_circuits import Observation
from prerequisite_circuits.demo import run
from prerequisite_circuits.state import evaluation, tensor_digest
from prerequisite_circuits.tapes import Batch, ReplayTape
from prerequisite_circuits.training import Branch, run_branches


def test_demo_removal_rescue_and_wrong_supply():
    results = run()
    assert len({r.metadata["tape_sha256"] for r in results.values()}) == 1
    assert len({r.metadata["initial_state_sha256"] for r in results.values()}) == 1
    for arm in ["available", "control", "supplied"]:
        assert results[arm].observations[-1].target_available >= 0.95
    for arm in ["suppressed", "wrong_supply"]:
        assert results[arm].observations[-1].target_available < 0.95
    assert (
        results["available"].metadata["endpoint_sha256"]
        == results["supplied"].metadata["endpoint_sha256"]
    )


def test_evaluation_restores_buffers_modes_and_randomness():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(0.5))
    model.train()
    model[0].eval()
    state = tensor_digest(model.state_dict())
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with evaluation(model):
        random.random()
        np.random.rand()
        torch.rand(3)
        model[0].running_mean.add_(10)
        assert not model.training
    assert tensor_digest(model.state_dict()) == state
    assert model.training and not model[0].training and model[1].training
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), cpu)


def test_parameter_mutation_in_observer_is_rejected():
    model = nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="modified"):
        with evaluation(model):
            model.weight.add_(1)


def test_observation_frequency_does_not_change_training():
    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(2, 1))
    tape = ReplayTape.from_batches([Batch(torch.ones(4, 2), torch.ones(4, 1)) for _ in range(4)])
    before = tensor_digest(model.state_dict())

    def observe(model, step):
        torch.rand(7)
        random.random()
        np.random.rand()
        return Observation(step, 0.5)

    kwargs = dict(
        optimizer=lambda m: torch.optim.AdamW(m.parameters(), lr=0.01),
        loss=lambda m, b: (m(b.inputs) - b.targets).square().mean(),
        observe=observe,
    )
    a = run_branches(model, tape, [Branch("a"), Branch("b")], eval_every=1, **kwargs)
    b = run_branches(model, tape, [Branch("a")], eval_every=4, **kwargs)
    assert (
        a["a"].metadata["endpoint_sha256"]
        == a["b"].metadata["endpoint_sha256"]
        == b["a"].metadata["endpoint_sha256"]
    )
    assert tensor_digest(model.state_dict()) == before


def test_mismatched_tape_is_rejected_before_training_second_branch():
    count = 0

    def factory():
        nonlocal count
        count += 1
        yield Batch(torch.full((1, 1), float(count)), torch.zeros(1, 1))

    with pytest.raises(ValueError, match="differs"):
        run_branches(
            nn.Linear(1, 1),
            ReplayTape(factory, 1),
            [Branch("a"), Branch("b")],
            optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
            loss=lambda m, b: m(b.inputs).square().mean(),
            observe=lambda m, s: Observation(s, 0.5),
        )


def test_batch_storage_is_not_mutable_through_a_replay():
    batch = Batch(torch.ones(1, 1), torch.zeros(1))
    tape = ReplayTape.from_batches([batch])
    batch.inputs.zero_()
    one = next(tape.factory())
    one.inputs.zero_()
    assert next(tape.factory()).inputs.item() == 1


def test_optimizer_cannot_accidentally_train_the_original_model():
    model = nn.Linear(1, 1)
    tape = ReplayTape.from_batches([Batch(torch.ones(1, 1), torch.ones(1, 1))])
    with pytest.raises(ValueError, match="this branch"):
        run_branches(
            model,
            tape,
            [Branch("a")],
            optimizer=lambda copied: torch.optim.SGD(model.parameters(), lr=0.1),
            loss=lambda m, b: m(b.inputs).square().mean(),
            observe=lambda m, step: Observation(step, 0.5),
        )
