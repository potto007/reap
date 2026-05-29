"""Gemma 4 (26B A4B) MoE helpers for REAP.

Gemma 4 differs from every other model REAP supports in three ways that this
module isolates into small, testable pieces:

1. **Fused experts** -- `Gemma4TextExperts` stores all experts as 3D parameter
   tensors `gate_up_proj` `(E, 2*I, H)` and `down_proj` `(E, H, I)` (like Llama-4),
   not a `ModuleList`. Its own `forward` does sparse dispatch and returns the
   *summed* routed output, so it cannot give REAP the per-expert, per-token
   activations its saliency criterion needs. `compute_fused_expert_activations`
   reproduces the per-expert MLP densely for all tokens.

2. **Inlined routing** -- there is no MoE-block submodule; `router` and `experts`
   are siblings on `Gemma4TextDecoderLayer`. The observer hooks the router and
   closes over the parent layer (see observer.py); this module only owns the math.

3. **Compound router** -- `Gemma4TextRouter` is not a bare `nn.Linear`; the expert
   projection is `router.proj` and there is an extra per-expert learnable scale
   `router.per_expert_scale` `(E,)`. `slice_gemma4_moe` prunes all of these
   consistently.

Everything here is pure tensor ops so it can be unit-tested without loading the
26B checkpoint or requiring transformers>=5.5.0.dev0.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch


@torch.no_grad()
def compute_fused_expert_activations(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    expert_input: torch.Tensor,
) -> torch.Tensor:
    """Compute each expert's output for every token (unweighted by the router).

    Mirrors the per-expert math in `Gemma4TextExperts.forward`:
        gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)
        out_e    = linear(act_fn(gate) * up, down_proj[e])
    but evaluates *all* experts over *all* tokens (no sparse dispatch, no router
    weighting), which is what REAP's activation-norm saliency requires.

    Looping over experts (rather than a single batched einsum) bounds peak memory
    to one expert's intermediate at a time plus the `(E, T, H)` accumulator -- the
    same memory profile as the loop-based path in `MoETransformerObserver`. For the
    26B model at long sequence length this is still heavy; use the layer-wise
    observer in that regime.

    Args:
        gate_up_proj: `(E, 2*I, H)` fused gate/up weights.
        down_proj:    `(E, H, I)` down-projection weights.
        act_fn:       activation (e.g. gelu_pytorch_tanh), applied to the gate half.
        expert_input: `(T, H)` already-normalized hidden states (the experts'
                      input inside the decoder layer).

    Returns:
        `(E, T, H)` per-expert outputs in the dtype/device of `expert_input`.
    """
    if expert_input.dim() != 2:
        raise ValueError(f"expert_input must be (T, H); got {tuple(expert_input.shape)}")
    num_experts, two_i, hidden = gate_up_proj.shape
    if two_i % 2 != 0:
        raise ValueError(f"gate_up_proj dim 1 must be even (2*I); got {two_i}")
    if down_proj.shape[0] != num_experts:
        raise ValueError(
            f"Expert count mismatch: gate_up_proj has {num_experts}, "
            f"down_proj has {down_proj.shape[0]}"
        )
    if expert_input.shape[-1] != hidden:
        raise ValueError(
            f"hidden mismatch: expert_input H={expert_input.shape[-1]} vs gate_up_proj H={hidden}"
        )

    total_tokens = expert_input.shape[0]
    activations = torch.zeros(
        (num_experts, total_tokens, hidden),
        device=expert_input.device,
        dtype=expert_input.dtype,
    )
    for e in range(num_experts):
        gate, up = torch.nn.functional.linear(expert_input, gate_up_proj[e]).chunk(2, dim=-1)
        hidden_e = act_fn(gate) * up
        activations[e] = torch.nn.functional.linear(hidden_e, down_proj[e])
    return activations


def router_probs_to_logits(router_probabilities: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Recover pre-softmax logits from full router probabilities.

    `Gemma4TextRouter.forward` returns the full softmax `router_probabilities`
    `(T, E)`, but REAP's `update_pruning_state` expects raw `router_logits` and
    applies softmax itself. Softmax is invariant to an additive constant, so
    `log(p)` is an exact inverse: `softmax(log(p)) == p`. The clamp guards the log
    against exact zeros from low-probability experts.

    Note: this intentionally reflects the *plain* router softmax, matching REAP's
    gate-value definition used for every other model. Gemma's `per_expert_scale`
    is applied to the routed weights downstream of softmax and is NOT folded into
    saliency here (it is still sliced correctly at prune time). Combined with
    `renormalize_router_weights=True`, the selected-expert weights REAP sees match
    Gemma's renormalized top-k weights before `per_expert_scale`.
    """
    return torch.log(router_probabilities.clamp_min(eps))


def _retained_index_tensor(
    retained_expert_indices: Sequence[int], device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(list(retained_expert_indices), dtype=torch.long, device=device)


@torch.no_grad()
def slice_gemma4_moe(decoder_layer, retained_expert_indices: Sequence[int]) -> int:
    """Prune a Gemma4 decoder layer's MoE in place to the retained experts.

    Slices, consistently:
      - `experts.gate_up_proj` `(E, 2I, H)` and `experts.down_proj` `(E, H, I)` rows,
      - `experts.num_experts`,
      - `router.proj.weight` `(E, H)` rows and `router.proj.out_features`,
      - `router.per_expert_scale` `(E,)`.

    The parallel dense `decoder_layer.mlp` (the always-on path summed with the MoE
    output) is left untouched -- REAP only prunes routed experts.

    Returns the number of retained experts.
    """
    experts = decoder_layer.experts
    router = decoder_layer.router
    device = experts.gate_up_proj.device
    idx = _retained_index_tensor(retained_expert_indices, device)
    n_retained = idx.numel()

    experts.gate_up_proj.data = experts.gate_up_proj.data.index_select(0, idx).contiguous()
    experts.down_proj.data = experts.down_proj.data.index_select(0, idx).contiguous()
    if hasattr(experts, "num_experts"):
        experts.num_experts = n_retained

    proj = router.proj
    proj.weight.data = proj.weight.data.index_select(0, idx.to(proj.weight.device)).contiguous()
    proj.out_features = n_retained
    if getattr(proj, "bias", None) is not None:
        proj.bias.data = proj.bias.data.index_select(0, idx.to(proj.bias.device)).contiguous()

    if hasattr(router, "per_expert_scale"):
        scale_idx = idx.to(router.per_expert_scale.device)
        router.per_expert_scale.data = (
            router.per_expert_scale.data.index_select(0, scale_idx).contiguous()
        )

    return n_retained
