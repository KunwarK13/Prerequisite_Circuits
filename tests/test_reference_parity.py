import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from prerequisite_circuits.interventions import NeoXHeads


@pytest.mark.artifact
def test_new_head_mask_matches_historical_forward_and_gradients():
    path = (
        Path(__file__).resolve().parents[1] / "artifacts/reference-source/code/core/pythia_clamp.py"
    )
    if not path.exists():
        pytest.skip("reference-source asset not downloaded")
    spec = importlib.util.spec_from_file_location("historical_clamp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = transformers.GPTNeoXConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        rotary_pct=0.5,
        attention_dropout=0,
        hidden_dropout=0,
    )
    config._attn_implementation = "eager"
    model = transformers.GPTNeoXForCausalLM(config).train()
    heads = [(0, 1), (1, 2)]
    inputs = torch.tensor([[1, 2, 3, 1], [3, 4, 2, 3]])
    original = module.TrainOnlyHeadClamp(model, heads)
    try:
        with original.on_target_examples():
            a = model(inputs).logits
            a.square().mean().backward()
        expected = {
            name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
        }
    finally:
        original.remove()
    model.zero_grad(set_to_none=True)
    with NeoXHeads(model, heads).apply():
        b = model(inputs).logits
        b.square().mean().backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], rtol=0, atol=0)
