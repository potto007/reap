"""End-to-end Gemma 4 REAP tests against a real (tiny) transformers model.

Unlike ``test_gemma4_fused.py`` (pure tensor math + fakes), these build an
actual ``Gemma4ForCausalLM`` from a small ``Gemma4TextConfig`` and exercise the
live wiring the 26B checkpoint would hit:

  * the ``Gemma4MoEExpertObserver`` hooking a real ``Gemma4TextRouter`` and
    unpacking its ``(router_probabilities, top_k_weights, top_k_index)`` output;
  * the expert-input self-check firing on the real ``experts`` module (a clean
    first pass is the validation that the router-hook input equals the experts'
    input -- see the observer's ``_verify_expert_input``);
  * ``compute_fused_expert_activations`` over the real fused parameter tensors;
  * a full ``prune()`` at ratio 0.5 that slices the fused experts + compound
    router via ``slice_gemma4_moe``, updates ``config.num_experts``, and writes a
    checkpoint that reloads smaller.

Skipped automatically if the installed transformers lacks Gemma 4 (needs
>=5.5.0). No checkpoint download: the model is randomly initialized and tiny.
"""

import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

Gemma4ForCausalLM = getattr(transformers, "Gemma4ForCausalLM", None)
Gemma4TextConfig = getattr(transformers, "Gemma4TextConfig", None)

pytestmark = pytest.mark.skipif(
    Gemma4ForCausalLM is None or Gemma4TextConfig is None,
    reason="installed transformers lacks Gemma 4 (needs >=5.5.0)",
)

NUM_EXPERTS = 8
TOP_K = 2


def _tiny_gemma4():
    """A minimal MoE Gemma 4 model. ``enable_moe_block=True`` is what creates the
    per-layer ``router``/``experts``/``pre_feedforward_layernorm_2`` the observer
    relies on; without it the decoder layers are dense."""
    cfg = Gemma4TextConfig(
        num_experts=NUM_EXPERTS,
        top_k_experts=TOP_K,
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=24,
        moe_intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        enable_moe_block=True,
    )
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg).eval(), cfg


def _run_forwards(model, cfg, *, n_batches, batch=2, seq=6, seed=1):
    torch.manual_seed(seed)
    for _ in range(n_batches):
        input_ids = torch.randint(0, cfg.vocab_size, (batch, seq))
        with torch.no_grad():
            model(input_ids)


def test_tiny_gemma4_is_moe():
    """Guards the fixture: a regression to dense layers would silently void the
    rest of this module."""
    model, _ = _tiny_gemma4()
    layer = model.model.layers[0]
    assert layer.enable_moe_block
    assert hasattr(layer, "router") and hasattr(layer, "experts")
    assert hasattr(layer, "pre_feedforward_layernorm_2")
    assert layer.experts.gate_up_proj.shape[0] == NUM_EXPERTS


def test_observer_hooks_real_gemma4_and_self_check_passes():
    from reap.observer import Gemma4MoEExpertObserver, Gemma4MoEObserverHookConfig

    model, cfg = _tiny_gemma4()
    observer = Gemma4MoEExpertObserver(model, Gemma4MoEObserverHookConfig())
    try:
        # One hook per MoE layer.
        assert len(observer.hooks) == cfg.num_hidden_layers

        _run_forwards(model, cfg, n_batches=3)

        # Every layer accumulated state and -- critically -- passed the
        # expert-input self-check. A raised RuntimeError here would mean the
        # router-hook input does NOT equal the experts' actual input, i.e. REAP
        # saliency would be scored on the wrong activations. A clean pass is the
        # real validation of that assumption against the installed transformers.
        assert set(observer.state.keys()) == {0, 1}
        assert observer._verified_layers == {0, 1}

        report = observer.report_state()
        for layer in (0, 1):
            assert report[layer]["total_tokens"] > 0
            # REAP saliency vector, one score per (pre-prune) expert.
            assert report[layer]["reap"].shape == (NUM_EXPERTS,)
            assert torch.isfinite(report[layer]["reap"]).all()
            assert report[layer]["expert_frequency"].shape == (NUM_EXPERTS,)
    finally:
        observer.close_hooks()


# NOTE: the self-check *failure* path (router-hook input != experts' actual input)
# is unit-tested directly in test_gemma4_fused.py::
# test_expert_input_self_check_fails_loudly_on_mismatch. It cannot be faithfully
# reproduced here by corrupting pre_feedforward_layernorm_2, because both the
# hook's recomputation and the experts' real input flow through the same module,
# so they diverge together and still match.


def test_prune_half_slices_experts_router_and_reloads(tmp_path):
    from reap.observer import Gemma4MoEExpertObserver, Gemma4MoEObserverHookConfig
    from reap.prune import prune

    model, cfg = _tiny_gemma4()
    observer = Gemma4MoEExpertObserver(model, Gemma4MoEObserverHookConfig())
    _run_forwards(model, cfg, n_batches=4, seq=8, seed=2)
    observer_data = observer.report_state()
    observer.close_hooks()

    prune_args = types.SimpleNamespace(
        prune_method="reap",
        perserve_super_experts=False,
        perserve_outliers=False,
    )
    n_prune = NUM_EXPERTS // 2  # prune half -> retain 4
    retained = NUM_EXPERTS - n_prune
    out_dir = tmp_path / "pruned"

    prune(
        observer_data,
        model,
        prune_args,
        n_experts_to_prune=n_prune,
        pruned_model_dir=out_dir,
    )

    # Config patched to the retained count.
    assert model.config.num_experts == retained
    assert cfg.top_k_experts <= retained  # routing still valid post-prune

    # Every component sliced consistently in every layer.
    for layer in model.model.layers:
        assert layer.experts.gate_up_proj.shape[0] == retained
        assert layer.experts.down_proj.shape[0] == retained
        assert layer.experts.num_experts == retained
        assert layer.router.proj.weight.shape[0] == retained
        assert layer.router.proj.out_features == retained
        assert layer.router.per_expert_scale.shape[0] == retained

    # The pruned in-memory model still runs a forward.
    with torch.no_grad():
        out = model(torch.randint(0, cfg.vocab_size, (1, 5)))
    assert out.logits.shape == (1, 5, cfg.vocab_size)

    # The saved checkpoint reloads as a smaller model.
    reloaded = Gemma4ForCausalLM.from_pretrained(out_dir)
    assert reloaded.config.num_experts == retained
    layer0 = reloaded.model.layers[0]
    assert layer0.experts.gate_up_proj.shape[0] == retained
    assert layer0.router.proj.weight.shape[0] == retained
    with torch.no_grad():
        reloaded(torch.randint(0, cfg.vocab_size, (1, 5)))


def test_setup_observer_selects_gemma_path_and_defaults_norm_topk_prob():
    """Regression for main._setup_observer's Gemma branch: it must pick the
    Gemma observer class/config and default norm_topk_prob to True (Gemma's
    config carries no such flag, but its router renormalizes by construction)."""
    from reap.main import _setup_observer
    from reap.observer import Gemma4MoEExpertObserver, Gemma4MoEObserverHookConfig

    model, _ = _tiny_gemma4()
    assert getattr(model.config, "norm_topk_prob", None) is None  # premise

    obs_args = types.SimpleNamespace(
        renormalize_router_weights=True,
        record_pruning_metrics_only=True,
    )
    observer = _setup_observer(model, obs_args)
    try:
        assert isinstance(observer, Gemma4MoEExpertObserver)
        assert isinstance(observer.hook_config, Gemma4MoEObserverHookConfig)
        # norm_topk_prob defaulted True for Gemma -> renormalization stays on.
        assert observer.hook_config.renormalize_router_weights is True
    finally:
        observer.close_hooks()


def test_setup_observer_renorm_off_when_obs_arg_disabled():
    from reap.main import _setup_observer
    from reap.observer import Gemma4MoEExpertObserver

    model, _ = _tiny_gemma4()
    obs_args = types.SimpleNamespace(
        renormalize_router_weights=False,
        record_pruning_metrics_only=True,
    )
    observer = _setup_observer(model, obs_args)
    try:
        assert isinstance(observer, Gemma4MoEExpertObserver)
        assert observer.hook_config.renormalize_router_weights is False
    finally:
        observer.close_hooks()
