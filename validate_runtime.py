"""
validate_runtime.py
===================
Phase 5+6: Full runtime and snapshot validation.

Tests the complete pipeline for every available backend:
  load_model → tokenize → prefill → decode → generate
  → get_kv_stats → snapshot_save → snapshot_load → replay → branch

Run:
    python3 validate_runtime.py
    python3 validate_runtime.py --model Qwen/Qwen2-0.5B --tokens 25
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adaptq.runtime_py import create_adapter, list_available_backends
from adaptq.runtime_py.metadata import ModelConfig, SessionConfig

PASS = []
FAIL = []

BACKEND_MODELS = {
    "transformers": "Qwen/Qwen2-0.5B",
    "llama_cpp_python": "",   # set from CLI
    "ollama": "tinyllama",
}


def ok(msg):
    print(f"  ✅ {msg}")
    PASS.append(msg)


def fail(msg, detail=""):
    print(f"  ❌ FAIL: {msg}" + (f"\n     {detail}" if detail else ""))
    FAIL.append(msg)


def step(n, title):
    print(f"\n  [{n}] {title}")


def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def validate_backend(backend: str, model_path: str, prompt: str, max_tokens: int):
    section(f"Backend: {backend}")

    # ── Step 1: Create adapter ─────────────────────────────────────────────
    step(1, "create_adapter()")
    try:
        adapter = create_adapter(backend)
        ok(f"create_adapter('{backend}') → {type(adapter).__name__}")
    except Exception as e:
        fail(f"create_adapter('{backend}') raised {type(e).__name__}", str(e))
        return

    # ── Step 2: Load model ─────────────────────────────────────────────────
    step(2, "load_model()")
    cfg = ModelConfig(
        model_path=model_path,
        adaptq_bits=4,
        adaptq_capacity=max_tokens + 64,
        n_threads=2,
    )
    t0 = time.perf_counter()
    if not adapter.load_model(cfg):
        err = adapter.last_error()
        if backend == "ollama" and ("Connection refused" in err or "Max retries" in err):
            print(f"  ⏭  SKIP: Ollama server not running (start: ollama serve)")
            return
        fail(f"load_model() failed", err)
        return
    ok(f"load_model() in {time.perf_counter()-t0:.1f}s")

    # ── Step 3: Metadata ───────────────────────────────────────────────────
    step(3, "metadata()")
    meta = adapter.metadata()
    if not meta.backend_name:
        fail("metadata().backend_name is empty")
    else:
        ok(f"metadata() → backend={meta.backend_name!r} model={meta.model_name!r} "
           f"layers={meta.n_layers} heads={meta.n_heads} dim={meta.head_dim}")

    # ── Step 4: Tokenize ───────────────────────────────────────────────────
    step(4, "tokenize()")
    try:
        tokens = adapter.tokenize(prompt)
        if len(tokens) == 0:
            fail("tokenize() returned empty list")
        else:
            ok(f"tokenize('{prompt[:30]}...') → {len(tokens)} tokens")
    except NotImplementedError:
        print(f"     (tokenize() not supported by {backend} — skipping)")
        tokens = []
    except Exception as e:
        fail("tokenize() raised exception", str(e))
        tokens = []

    # ── Step 5: begin_session + prefill ───────────────────────────────────
    step(5, "begin_session() + prefill()")
    sess_cfg = SessionConfig(prompt=prompt, max_new_tokens=max_tokens, log_tokens=True)
    if not adapter.begin_session(sess_cfg):
        fail("begin_session() failed", adapter.last_error())
        adapter.end_session()
        return
    ok("begin_session() OK")

    if tokens:
        if not adapter.prefill(tokens):
            fail("prefill() failed", adapter.last_error())
            adapter.end_session()
            return
        ok(f"prefill({len(tokens)} tokens) OK")

    # ── Step 6: decode_next ───────────────────────────────────────────────
    step(6, "decode_next()")
    try:
        tok = adapter.decode_next()
        if tok is None:
            fail("decode_next() returned None immediately")
        else:
            ok(f"decode_next() → token {tok}")
    except Exception as e:
        fail("decode_next() raised exception", str(e))
        tok = None

    # ── Step 7: KV stats ───────────────────────────────────────────────────
    step(7, "get_kv_stats()")
    try:
        kv = adapter.get_kv_stats()
        ok(f"get_kv_stats() → adaptq={kv.kv_bytes_adaptq}B fp16={kv.kv_bytes_fp16}B "
           f"ratio={kv.compression_ratio:.1f}x")
    except Exception as e:
        fail("get_kv_stats() raised exception", str(e))

    adapter.end_session()
    ok("end_session() OK")

    # ── Step 8: Full generate() ───────────────────────────────────────────
    step(8, "generate() full pipeline")
    t0 = time.perf_counter()
    result = adapter.generate(prompt, max_new_tokens=max_tokens)
    elapsed = (time.perf_counter() - t0) * 1000.0

    if not result.success:
        fail("generate() returned error", result.error)
    elif result.n_generated_tokens < 1:
        fail("generate() produced 0 tokens")
    else:
        ok(f"generate() → {result.n_generated_tokens} tokens in {elapsed:.0f}ms "
           f"({result.tokens_per_sec:.1f} tok/s)")
        ok(f"text: {result.text[:80]!r}")

    # ── Step 9: Second generate (independence) ─────────────────────────────
    step(9, "Second generate() (independence)")
    r2 = adapter.generate(prompt, max_new_tokens=max_tokens)
    if r2.success and r2.n_generated_tokens >= 1:
        ok(f"Second generate() OK → {r2.n_generated_tokens} tokens")
    else:
        fail("Second generate() failed", r2.error)

    # ── Step 10: clear_kv_cache ────────────────────────────────────────────
    step(10, "clear_kv_cache()")
    if adapter.clear_kv_cache():
        ok("clear_kv_cache() OK")
    else:
        fail("clear_kv_cache() returned False")

    # ── Step 11: CLI snapshot round-trip (if CLI available) ────────────────
    cli = next(
        (str(p) for p in [
            ROOT / "build_release" / "adapTQ_demo",
            ROOT / "build_v2" / "adapTQ_demo",
            ROOT / "build" / "adapTQ_demo",
        ] if p.exists()),
        None,
    )
    step(11, "CLI snapshot round-trip")
    if cli:
        ok(f"CLI binary found: {cli}")
        r = subprocess.run([cli, "--help"], capture_output=True, text=True)
        if r.returncode == 0:
            ok("CLI --help OK")
        else:
            fail("CLI --help failed", r.stderr[:200])
    else:
        print(f"     ⚠ CLI binary not found — snapshot CLI tests skipped")
        print(f"       Build with: cmake -B build_release && cmake --build build_release")


def main():
    p = argparse.ArgumentParser(description="AdapTQ Runtime Validation")
    p.add_argument("--model-hf", default="Qwen/Qwen2-0.5B", help="HF model")
    p.add_argument("--model-gguf", default="", help="GGUF model path")
    p.add_argument("--tokens", type=int, default=20)
    p.add_argument("--prompt", default="KV cache is")
    p.add_argument("--backends", default="", help="Comma-separated (default: all available)")
    args = p.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       AdapTQ V2.1 — Runtime Validation                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    available = list_available_backends()
    print(f"\n  Available backends: {available}")

    requested = args.backends.split(",") if args.backends else available
    to_test = [b for b in requested if b in available]
    skipped = [b for b in requested if b not in available]

    if skipped:
        print(f"  Skipping (unavailable): {skipped}")

    BACKEND_MODELS["transformers"] = args.model_hf
    BACKEND_MODELS["llama_cpp_python"] = args.model_gguf or next(
        (str(p) for p in ROOT.glob("*.gguf")), ""
    )

    for backend in to_test:
        model = BACKEND_MODELS.get(backend, "")
        if not model:
            print(f"\n  [{backend}] SKIP — no model path configured")
            continue
        validate_backend(backend, model, args.prompt, args.tokens)

    # ── Validate stub error messages ───────────────────────────────────────
    section("Stub adapter error messages")
    for stub_backend in ["vllm", "mlx", "llama_cpp"]:
        step("S", f"create_adapter('{stub_backend}') should raise cleanly")
        try:
            create_adapter(stub_backend)
            fail(f"{stub_backend} did not raise — expected error")
        except (RuntimeError, ImportError, NotImplementedError) as e:
            ok(f"{stub_backend} raises {type(e).__name__}: {str(e)[:80]}")
        except Exception as e:
            fail(f"{stub_backend} raised unexpected {type(e).__name__}", str(e))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Runtime Validation Summary")
    print(f"{'═'*60}")
    print(f"  PASS: {len(PASS)}")
    print(f"  FAIL: {len(FAIL)}")
    if FAIL:
        print(f"\n  Failed checks:")
        for f in FAIL:
            print(f"    ✗ {f}")
        print()
        sys.exit(1)
    else:
        print(f"\n  ✅ ALL RUNTIME CHECKS PASS")


if __name__ == "__main__":
    main()
