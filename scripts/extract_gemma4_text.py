"""Extract a text-only Gemma4ForCausalLM checkpoint from the Gemma 4 26B A4B VLM.

Gemma 4 26B A4B ships as `Gemma4ForConditionalGeneration` (a vision-language
model). REAP only prunes the routed experts in the *text* decoder, so the vision
tower, audio tower, and multimodal embedders are dead weight during calibration
and they break REAP's `model.model.layers[...]` assumption.

`transformers` already provides a text-only `Gemma4ForCausalLM` that wraps the
exact same `Gemma4TextModel` submodule and `Gemma4TextConfig` as the VLM, so the
text weights transfer bit-for-bit (no conversion loss).

This script copies the text weights at the *tensor* level instead of through a
live model object. The 26B VLM is ~52GB in bf16; the text decoder alone is larger
than a typical workstation's RAM, and loading it through `from_pretrained`
(even with `device_map="auto"`) makes accelerate offload the overflow to the meta
device, which then has no data to save. Streaming the shards sidesteps all of
that: peak memory is one output shard (a few GB), and no GPU is required.

  1. resolve the cached source snapshot + its safetensors index,
  2. select every `model.language_model.*` tensor and re-prefix it to `model.*`,
  3. stream those tensors shard-by-shard into a fresh sharded checkpoint,
  4. write a `Gemma4ForCausalLM` config (architectures + norm_topk_prob), the
     generation config, and the tokenizer.

The result loads via plain `AutoModelForCausalLM` and exposes `model.model.layers`,
which is what REAP's `model_util.get_moe()` expects.

NOTE: Gemma 4 requires transformers >= 5.5.0.dev0. Run this in an env with a
recent transformers; the produced checkpoint needs the same version to be loaded
later by REAP.

Usage:
    python scripts/extract_gemma4_text.py \
        --src google/gemma-4-26B-A4B-it \
        --out artifacts/models/gemma-4-26B-A4B-text \
        [--dtype bfloat16] [--shard-size-gb 5] [--smoke-test]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_gemma4_text")

# The VLM nests the text stack as model.language_model.*; the text-only class
# wraps the identical module as self.model, i.e. model.*.
_SRC_PREFIX = "model.language_model."
_DST_PREFIX = "model."


def _require_gemma4() -> None:
    try:
        from transformers import Gemma4ForCausalLM  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "Could not import Gemma4 classes from transformers. Gemma 4 requires "
            "transformers>=5.5.0.dev0. Run this in an env with a recent transformers.\n"
            f"Original error: {e}"
        )


def _resolve_snapshot(src: str) -> pathlib.Path:
    """Return a local dir containing model.safetensors.index.json + shards."""
    local = pathlib.Path(src)
    if (local / "model.safetensors.index.json").exists():
        return local
    from huggingface_hub import snapshot_download

    return pathlib.Path(
        snapshot_download(
            src,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "tokenizer*",
                "*.model",
                "chat_template*",
            ],
        )
    )


def _build_text_config(src: str):
    """Derive a standalone Gemma4ForCausalLM config from the VLM's text_config."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(src)
    text_cfg = cfg.text_config
    text_cfg.architectures = ["Gemma4ForCausalLM"]
    # Gemma's router renormalizes its top-k weights by construction; surface this as
    # norm_topk_prob so REAP's observer enables top-k weight renormalization (matching
    # the renorm REAP applies for every other model).
    text_cfg.norm_topk_prob = True
    return text_cfg


def extract(src: str, out: pathlib.Path, dtype: str, shard_size_gb: float = 5.0) -> pathlib.Path:
    from safetensors import safe_open
    from safetensors.torch import save_file

    _require_gemma4()
    out_dtype = None if dtype == "auto" else getattr(torch, dtype)

    snap = _resolve_snapshot(src)
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    # Select the text tensors and compute their destination keys.
    selected = [
        (_DST_PREFIX + k[len(_SRC_PREFIX):], k, shard)
        for k, shard in weight_map.items()
        if k.startswith(_SRC_PREFIX)
    ]
    if not selected:
        raise RuntimeError(
            f"No {_SRC_PREFIX}* tensors found in {snap}/model.safetensors.index.json"
        )
    logger.info("Selected %d text tensors out of %d total.", len(selected), len(weight_map))

    text_cfg = _build_text_config(src)
    tied = bool(getattr(text_cfg, "tie_word_embeddings", True))
    logger.info(
        "Text config: %d layers, %d experts/layer, top_k=%s, vocab=%d, tie_word_embeddings=%s",
        text_cfg.num_hidden_layers,
        text_cfg.num_experts,
        getattr(text_cfg, "top_k_experts", "?"),
        text_cfg.vocab_size,
        tied,
    )

    out.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}

    def handle(shard: str):
        if shard not in handles:
            handles[shard] = safe_open(str(snap / shard), framework="pt", device="cpu")
        return handles[shard]

    shard_limit = int(shard_size_gb * (1024 ** 3))
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    total_size = 0
    written: list[tuple[pathlib.Path, list[str]]] = []
    embed_key = _DST_PREFIX + "embed_tokens.weight"
    embed_for_lm_head: torch.Tensor | None = None

    def flush() -> None:
        nonlocal buf, buf_bytes
        if not buf:
            return
        tmp = out / f".shard-{len(written) + 1}.safetensors"
        save_file(buf, str(tmp), metadata={"format": "pt"})
        written.append((tmp, list(buf.keys())))
        logger.info("Wrote %s (%d tensors, %.2f GB).", tmp.name, len(buf), buf_bytes / 1e9)
        buf = {}
        buf_bytes = 0

    for dst_key, src_key, shard in selected:
        t = handle(shard).get_tensor(src_key)
        if out_dtype is not None and t.dtype != out_dtype:
            t = t.to(out_dtype)
        if dst_key == embed_key and not tied:
            embed_for_lm_head = t  # keep a ref to clone into lm_head.weight below
        nbytes = t.numel() * t.element_size()
        buf[dst_key] = t
        buf_bytes += nbytes
        total_size += nbytes
        if buf_bytes >= shard_limit:
            flush()

    # If embeddings are not tied, the text-only model needs an explicit lm_head.
    # (Gemma ties them, so this branch is normally inert.)
    if not tied:
        if embed_for_lm_head is None:
            raise RuntimeError("tie_word_embeddings is False but embed_tokens.weight was not found.")
        lm = embed_for_lm_head.clone()
        buf["lm_head.weight"] = lm
        buf_bytes += lm.numel() * lm.element_size()
        total_size += lm.numel() * lm.element_size()
    flush()

    # Rename temp shards to the canonical model-XXXXX-of-YYYYY.safetensors and build
    # the weight map now that the shard count is known.
    n = len(written)
    out_weight_map: dict[str, str] = {}
    for i, (tmp, keys) in enumerate(written, 1):
        final = out / f"model-{i:05d}-of-{n:05d}.safetensors"
        tmp.rename(final)
        for k in keys:
            out_weight_map[k] = final.name

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": out_weight_map}, indent=2)
    )
    logger.info("Wrote %d shards, %.2f GB total.", n, total_size / 1e9)

    text_cfg.save_pretrained(out)
    _save_generation_config(snap, out)
    _save_tokenizer(src, out)
    _verify(out, text_cfg, out_weight_map)

    logger.info("Done. architectures=%s", text_cfg.architectures)
    return out


def _verify(out: pathlib.Path, text_cfg, weight_map: dict[str, str]) -> None:
    from safetensors import safe_open

    assert weight_map.get(_DST_PREFIX + "embed_tokens.weight"), "Missing embed_tokens.weight"
    gate_key = _DST_PREFIX + "layers.0.experts.gate_up_proj"
    shard = weight_map.get(gate_key)
    assert shard, f"Missing {gate_key}"
    with safe_open(str(out / shard), framework="pt", device="cpu") as f:
        shape = f.get_slice(gate_key).get_shape()
    assert shape[0] == text_cfg.num_experts, (
        f"Expert count mismatch: gate_up_proj[0]={shape[0]} != num_experts={text_cfg.num_experts}"
    )
    logger.info("Verified: layers.0.experts.gate_up_proj=%s (num_experts=%d).", tuple(shape), text_cfg.num_experts)


def _save_generation_config(snap: pathlib.Path, out: pathlib.Path) -> None:
    src = snap / "generation_config.json"
    if src.exists():
        (out / "generation_config.json").write_text(src.read_text())
        logger.info("Copied generation_config.json.")


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

    logger.info("Smoke test: reloading %s via AutoModelForCausalLM (device_map=%s)...", out, device)
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
        help="Output dtype. 'auto' preserves the stored dtype.",
    )
    p.add_argument(
        "--shard-size-gb",
        type=float,
        default=5.0,
        help="Approximate max size of each output safetensors shard. Bounds peak RAM.",
    )
    p.add_argument("--smoke-test", action="store_true", help="Reload + generate after saving.")
    p.add_argument("--smoke-device", default="auto", help="device_map for smoke test (auto/cpu/cuda).")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    extract(args.src, out, args.dtype, shard_size_gb=args.shard_size_gb)
    if args.smoke_test:
        smoke_test(out, args.smoke_device)


if __name__ == "__main__":
    main()
