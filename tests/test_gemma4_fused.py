import types

import pytest
import torch
import torch.nn as nn

from reap.models.gemma4 import (
    compute_fused_expert_activations,
    router_probs_to_logits,
    slice_gemma4_moe,
)
from reap.observer import Gemma4MoEExpertObserver


def _reference_expert_activations(gate_up_proj, down_proj, act_fn, x):
    """Independent einsum reference for the per-expert MLP (all experts, all tokens)."""
    gu = torch.einsum("th,eoh->eto", x, gate_up_proj)  # (E, T, 2I)
    gate, up = gu.chunk(2, dim=-1)
    inter = act_fn(gate) * up  # (E, T, I)
    return torch.einsum("eti,ehi->eth", inter, down_proj)  # (E, T, H)


def test_compute_fused_expert_activations_matches_reference():
    torch.manual_seed(0)
    E, I, H, T = 5, 7, 6, 11
    gate_up_proj = torch.randn(E, 2 * I, H)
    down_proj = torch.randn(E, H, I)
    x = torch.randn(T, H)
    act_fn = nn.GELU(approximate="tanh")  # gemma uses gelu_pytorch_tanh

    got = compute_fused_expert_activations(gate_up_proj, down_proj, act_fn, x)
    ref = _reference_expert_activations(gate_up_proj, down_proj, act_fn, x)

    assert got.shape == (E, T, H)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_compute_fused_expert_activations_matches_gemma_per_expert_forward():
    """Expert 0's dense output must equal Gemma's own gate/up/down composition."""
    torch.manual_seed(1)
    E, I, H, T = 3, 4, 5, 8
    gate_up_proj = torch.randn(E, 2 * I, H)
    down_proj = torch.randn(E, H, I)
    x = torch.randn(T, H)
    act_fn = nn.GELU(approximate="tanh")

    acts = compute_fused_expert_activations(gate_up_proj, down_proj, act_fn, x)

    gate, up = torch.nn.functional.linear(x, gate_up_proj[0]).chunk(2, dim=-1)
    expected0 = torch.nn.functional.linear(act_fn(gate) * up, down_proj[0])
    torch.testing.assert_close(acts[0], expected0)


def test_router_probs_to_logits_roundtrip():
    torch.manual_seed(2)
    probs = torch.softmax(torch.randn(13, 9), dim=-1)
    logits = router_probs_to_logits(probs)
    torch.testing.assert_close(torch.softmax(logits, dim=-1), probs, rtol=1e-5, atol=1e-6)


class _FakeExperts(nn.Module):
    def __init__(self, E, I, H):
        super().__init__()
        self.num_experts = E
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.randn(E, H, I))
        self.act_fn = nn.GELU(approximate="tanh")


class _FakeRouter(nn.Module):
    def __init__(self, E, H):
        super().__init__()
        self.proj = nn.Linear(H, E, bias=False)
        self.per_expert_scale = nn.Parameter(torch.randn(E))


class _FakeDecoderLayer(nn.Module):
    def __init__(self, E, I, H):
        super().__init__()
        self.experts = _FakeExperts(E, I, H)
        self.router = _FakeRouter(E, H)


def test_slice_gemma4_moe_prunes_all_components_consistently():
    torch.manual_seed(3)
    E, I, H = 8, 4, 6
    layer = _FakeDecoderLayer(E, I, H)

    gate_up_before = layer.experts.gate_up_proj.detach().clone()
    down_before = layer.experts.down_proj.detach().clone()
    proj_before = layer.router.proj.weight.detach().clone()
    scale_before = layer.router.per_expert_scale.detach().clone()

    retained = [1, 3, 4, 7]  # prune 0,2,5,6
    n = slice_gemma4_moe(layer, retained)
    idx = torch.tensor(retained)

    assert n == len(retained)
    assert layer.experts.num_experts == len(retained)
    assert layer.experts.gate_up_proj.shape == (len(retained), 2 * I, H)
    assert layer.experts.down_proj.shape == (len(retained), H, I)
    assert layer.router.proj.weight.shape == (len(retained), H)
    assert layer.router.proj.out_features == len(retained)
    assert layer.router.per_expert_scale.shape == (len(retained),)

    # Retained rows must be exactly the originals, in order.
    torch.testing.assert_close(layer.experts.gate_up_proj.data, gate_up_before[idx])
    torch.testing.assert_close(layer.experts.down_proj.data, down_before[idx])
    torch.testing.assert_close(layer.router.proj.weight.data, proj_before[idx])
    torch.testing.assert_close(layer.router.per_expert_scale.data, scale_before[idx])


def _bare_gemma_observer():
    """A Gemma4MoEExpertObserver without running __init__ (which needs a real model)."""
    obs = Gemma4MoEExpertObserver.__new__(Gemma4MoEExpertObserver)
    obs._verified_layers = set()
    return obs


def test_expert_input_self_check_passes_when_input_matches():
    obs = _bare_gemma_observer()
    experts = nn.Identity()  # pre-hook fires when called
    layer = types.SimpleNamespace(experts=experts)
    expected = torch.randn(4, 6)

    obs._verify_expert_input(layer, layer_number=0, expert_input=expected)
    _ = experts(expected)  # same tensor the router hook computed

    assert 0 in obs._verified_layers


def test_expert_input_self_check_fails_loudly_on_mismatch():
    obs = _bare_gemma_observer()
    experts = nn.Identity()
    layer = types.SimpleNamespace(experts=experts)
    expected = torch.randn(4, 6)

    obs._verify_expert_input(layer, layer_number=1, expert_input=expected)
    with pytest.raises(RuntimeError, match="expert-input check FAILED"):
        experts(expected + 1.0)  # diverged input -> must raise

    assert 1 not in obs._verified_layers
