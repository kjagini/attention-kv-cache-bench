"""Run benchmark_all_attention.py on a Modal A100.

Usage (after `modal token new` or with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set):

    modal run modal_attention_bench.py
    modal run modal_attention_bench.py --hourly-usd 4.00

The benchmark code runs unchanged; only the device and hardware differ.
"""

import contextlib
import io

import modal

app = modal.App("attention-bench")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
    .add_local_python_source(
        "multi_head_attention",
        "kv_cache_mha",
        "mqa_kv_cache",
        "gqa_kv_cache",
        "mla_kv_cache",
        "cost_model",
        "benchmark_all_attention",
    )
)

SMALL = dict(d_model=512, num_heads=8, num_kv_heads=2, latent_dim=64)
LARGE = dict(d_model=4096, num_heads=32, num_kv_heads=8, latent_dim=512)

CONFIGS_SMALL = [
    dict(batch=1, prompt_len=128, gen_len=64, **SMALL),
    dict(batch=4, prompt_len=256, gen_len=64, **SMALL),
]
# Larger model + longer contexts, where KV-cache traffic should start to matter.
CONFIGS_LARGE = [
    dict(batch=1, prompt_len=2048, gen_len=64, **LARGE),
    dict(batch=8, prompt_len=2048, gen_len=64, **LARGE),
    dict(batch=1, prompt_len=8192, gen_len=64, **LARGE),
    dict(batch=4, prompt_len=4096, gen_len=64, **LARGE),
]


@app.function(gpu="A100", image=image, timeout=1800)
def bench(batch: int, prompt_len: int, gen_len: int, d_model: int, num_heads: int,
          num_kv_heads: int, latent_dim: int, hourly_usd: float) -> str:
    import argparse

    import torch
    import benchmark_all_attention as b

    args = argparse.Namespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch=batch,
        prompt_len=prompt_len,
        gen_len=gen_len,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        latent_dim=latent_dim,
        warmup=2,
        repeats=5,
        hourly_usd=hourly_usd,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
        b.run(args)
    return buf.getvalue()


@app.local_entrypoint()
def main(hourly_usd: float = 2.50, large: bool = True):
    for cfg in (CONFIGS_LARGE if large else CONFIGS_SMALL):
        print(bench.remote(hourly_usd=hourly_usd, **cfg))
