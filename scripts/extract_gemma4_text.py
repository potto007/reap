"""Extract a text-only Gemma4ForCausalLM checkpoint from the Gemma 4 26B A4B VLM.

Gemma 4 26B A4B ships as `Gemma4ForConditionalGeneration` (a vision-language
model). REAP only prunes the routed experts in the *text* decoder, so the vision
tower, audio tower, and multimodal embedders are dead weight during calibration
and they break REAP's `model.model.layers[...]` assumption.

`transformers` already provides a text-only `Gemma4ForCausalLM` that wraps the
exact same `Gemma4TextModel` submodule and `Gemma4TextConfig` as the VLM, so the
text weights transfer bit-for-bit (no conversion loss). This script:

  1. loads the VLM (CPU, low-mem),
  2. instantiates an empty `Gemma4ForCausalLM` on the meta device,
  3. assign-loads the VLM's `model.language_model.*` weights into it
     (re-prefixed to `model.*`) plus the tied `lm_head`,
  4. saves a standard checkpoint with `architectures: ["Gemma4ForCausalLM"]`.

The result loads via plain `AutoModelForCausalLM` and exposes `model.model.layers`,
which is what REAP's `model_util.get_moe()` expects.

NOTE: Gemma 4 requires transformers >= 5.5.0.dev0. The repo currently pins
transformers==4.55.0, so run this in an isolated environment with a newer
transformers (e.g. `uv run --with 'transformers>=5.5.0.dev0' python
scripts/extract_gemma4_text.py`). The produced checkpoint still needs the same
transformers version to be loaded later by REAP.

Usage:
    python scripts/extract_gemma4_text.py \
        --src google/gemma-4-26B-A4B-it \
        --out artifacts/models/gemma-4-26B-A4B-text \
        [--dtype bfloat16] [--smoke-test]
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_gemma4_text")


def _import_gemma4():
    try:
        from transformers import (  # type: ignore
            Gemma4ForCausalLM,
            Gemma4ForConditionalGeneration,
        )
    except ImportError as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "Could not import Gemma4 classes from transformers. Gemma 4 requires "
            "transformers>=5.5.0.dev0; the repo pins transformers==4.55.0. Run this "
            "script in an isolated env, e.g.:\n"
            "  uv run --with 'transformers>=5.5.0.dev0' python scripts/extract_gemma4_text.py ...\n"
            f"Original error: {e}"
        )
    return Gemma4ForConditionalGeneration, Gemma4ForCausalLM


def extract(src: str, out: pathlib.Path, dtype: str) -> pathlib.Path:
    from accelerate import init_empty_weights

    Gemma4ForConditionalGeneration, Gemma4ForCausalLM = _import_gemma4()

    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)

    logger.info("Loading VLM %s on CPU (low_cpu_mem_usage)...", src)
    vlm = Gemma4ForConditionalGeneration.from_pretrained(
        src,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    text_config = vlm.config.text_config

    # The VLM nests the text stack as model.language_model (a Gemma4TextModel);
    # the text-only class wraps the identical module as self.model.
    language_model = vlm.model.language_model
    logger.info(
        "Text stack: %s with %d layers, %d experts/layer, top_k=%d",
        language_model.__class__.__name__,
        text_config.num_hidden_layers,
        text_config.num_experts,
        getattr(text_config, "top_k_experts", -1),
    )

    # Build the destination model without allocating real storage, then
    # assign-load so we don't hold two full copies of the 26B weights in RAM.
    logger.info("Instantiating empty Gemma4ForCausalLM on meta device...")
    with init_empty_weights():
        text_model = Gemma4ForCausalLM(text_config)

    # Re-prefix language_model.* -> model.* and graft the tied lm_head.
    src_sd = language_model.state_dict()
    new_sd = {f"model.{k}": v for k, v in src_sd.items()}
    new_sd["lm_head.weight"] = vlm.lm_head.weight

    logger.info("Assign-loading %d tensors into text model...", len(new_sd))
    missing, unexpected = text_model.load_state_dict(new_sd, strict=False, assign=True)

    # The only acceptable "missing" keys are non-persistent buffers that both
    # classes regenerate (rotary inv_freq etc.). Anything in model.* that is
    # missing means a real prefix/shape mismatch -> fail loudly.
    hard_missing = [k for k in missing if k.startswith("model.") and k.endswith(".weight")]
    hard_missing += [k for k in missing if "experts" in k or "router" in k]
    if hard_missing:
        raise RuntimeError(f"Unexpected missing weight keys after load: {hard_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected extra keys after load: {unexpected}")
    if missing:
        logger.info("Ignored %d non-persistent buffer keys: %s", len(missing), missing[:5])

    # Sanity check: a representative fused-expert + router tensor survived.
    l0 = text_model.model.layers[0]
    assert hasattr(l0, "experts") and hasattr(l0.experts, "gate_up_proj"), (
        "Expected fused experts (gate_up_proj) on decoder layer 0."
    )
    assert l0.experts.gate_up_proj.shape[0] == text_config.num_experts, (
        f"Expert count mismatch: {l0.experts.gate_up_proj.shape[0]} != {text_config.num_experts}"
    )
    assert hasattr(l0.router, "proj") and hasattr(l0.router, "per_expert_scale"), (
        "Expected compound router with .proj and .per_expert_scale."
    )
    logger.info(
        "Verified layer 0: gate_up_proj=%s, down_proj=%s, router.proj.weight=%s, per_expert_scale=%s",
        tuple(l0.experts.gate_up_proj.shape),
        tuple(l0.experts.down_proj.shape),
        tuple(l0.router.proj.weight.shape),
        tuple(l0.router.per_expert_scale.shape),
    )

    # Gemma's router renormalizes its top-k weights by construction; surface this
    # as `norm_topk_prob` so REAP's observer enables top-k weight renormalization
    # (matching the renorm REAP applies for every other model).
    text_model.config.norm_topk_prob = True

    out.mkdir(parents=True, exist_ok=True)
    logger.info("Saving text-only checkpoint to %s ...", out)
    text_model.save_pretrained(out, safe_serialization=True)

    # Carry over the tokenizer (and processor, if present, harmlessly).
    _save_tokenizer(src, out)

    logger.info("Done. architectures=%s", text_model.config.architectures)
    return out


def _save_tokenizer(src: str, out: pathlib.Path) -> None:
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(src)
        tok.save_pretrained(out)
        logger.info("Saved tokenizer.")
    except Exception as e:  # pragma: no cover
        logger.warning("Could not save tokenizer (%s); copy it manually if needed.", e)


def smoke_test(out: pathlib.Path, device: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Smoke test: reloading %s via AutoModelForCausalLM...", out)
    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForCausalLM.from_pretrained(out, dtype="auto", device_map=device)
    model.eval()
    inputs = tok("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    logger.info("Smoke-test generation: %r", text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--out", default="artifacts/models/gemma-4-26B-A4B-text")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument("--smoke-test", action="store_true", help="Reload + generate after saving (slow on CPU).")
    p.add_argument("--smoke-device", default="auto", help="device_map for smoke test (auto/cpu/cuda).")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    extract(args.src, out, args.dtype)
    if args.smoke_test:
        smoke_test(out, args.smoke_device)


if __name__ == "__main__":
    main()
