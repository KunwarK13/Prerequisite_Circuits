import pytest

torch = pytest.importorskip("torch")
from torch import nn

from prerequisite_circuits.interventions import Channels, NeoXHeads, capture


def test_scope_preserves_support_rows_and_nonselected_gradients():
    layer = nn.Identity()
    x = torch.ones(2, 3, 4, requires_grad=True)
    with Channels(layer, [1], location="output").apply(scope=torch.tensor([True, False])):
        y = layer(x)
        y.sum().backward()
    assert torch.equal(y[1], x[1])
    assert torch.count_nonzero(y[0, :, 1]) == 0
    expected = torch.ones_like(x)
    expected[0, :, 1] = 0
    torch.testing.assert_close(x.grad, expected)
    torch.testing.assert_close(layer(x), x)


def test_supply_replaces_only_selected_channels_and_detaches_donor():
    layer = nn.Identity()
    x = torch.ones(2, 3, requires_grad=True)
    donor = torch.full_like(x, 7.0, requires_grad=True)
    with Channels(layer, [1], location="output").apply(replacement=donor):
        y = layer(x)
        y.sum().backward()
    torch.testing.assert_close(y, torch.tensor([[1.0, 7.0, 1.0], [1.0, 7.0, 1.0]]))
    assert donor.grad is None
    torch.testing.assert_close(x.grad, torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))


def test_optional_donor_gradient_path_is_explicit():
    layer = nn.Identity()
    donor = torch.ones(1, 2, requires_grad=True)
    with Channels(layer, [0], location="output").apply(replacement=donor, detach_replacement=False):
        layer(torch.zeros(1, 2)).sum().backward()
    torch.testing.assert_close(donor.grad, torch.tensor([[1.0, 0.0]]))


def test_input_hook_cleanup_after_exception_and_nested_rejection():
    layer = nn.Linear(2, 1, bias=False)
    intervention = Channels(layer, [0])
    with pytest.raises(RuntimeError, match="nested"):
        with intervention.apply():
            with intervention.apply():
                pass
    assert not layer._forward_pre_hooks
    with pytest.raises(ValueError):
        with intervention.apply(scope=torch.tensor([True, False])):
            layer(torch.ones(3, 2))
    assert not layer._forward_pre_hooks


def test_token_scope_and_capture_are_independent():
    layer = nn.Identity()
    x = torch.ones(2, 3, 4)
    scope = torch.tensor([[True, False, False], [False, False, True]])
    with capture(layer) as captured:
        with Channels(layer, [0], location="output").apply(scope=scope):
            y = layer(x)
    assert torch.equal(captured[0], x)
    assert y[0, 0, 0] == 0 and y[1, 2, 0] == 0 and y[0, 1, 0] == 1
    x.zero_()
    assert captured[0].sum() == 24
    assert not layer._forward_hooks and not layer._forward_pre_hooks


@pytest.mark.integration
def test_neox_channels_match_attention_head_contributions():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPTNeoXConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        rotary_pct=0.5,
        attention_dropout=0,
        hidden_dropout=0,
    )
    config._attn_implementation = "eager"
    model = transformers.GPTNeoXForCausalLM(config).eval()
    attention = model.gpt_neox.layers[0].attention
    with torch.no_grad(), capture(attention.query_key_value, location="output") as qkv:
        with NeoXHeads(model, [(0, 1)]).apply():
            with capture(attention.dense) as inputs:
                result = model(torch.tensor([[1, 2, 3, 1]]), output_attentions=True)
    values = qkv[0].view(1, 4, 4, 24).transpose(1, 2)[..., 16:]
    expected = (result.attentions[0] @ values).transpose(1, 2).reshape(1, 4, 32)
    expected[..., 8:16] = 0
    torch.testing.assert_close(inputs[0], expected, rtol=1e-5, atol=1e-6)
    assert not attention.dense._forward_pre_hooks
