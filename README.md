# AdapTQ: Adaptive Streaming Vector Quantization

**AdapTQ** is a production-grade C++17 KV cache quantization engine for LLM inference on edge and memory-constrained systems. 

**The Integration Pitch:** AdapTQ is an **optional KV-cache backend**. It runs entirely on the CPU, requires **no model changes**, and fits into existing inference pipelines with minimal adapter-style wrapper logic.

## 🚀 Quickstart

```python
import torch; from adaptq import AdaptQAttention
# 1. Initialize drop-in PyTorch wrapper (4-bit default)
layer = AdaptQAttention(dim=128, heads=4)
# 2. Forward pass dynamically routes continuous BxHxD generation tensors
out = layer(q=torch.randn(1, 4, 128), k=torch.randn(1, 4, 128), v=torch.randn(1, 4, 128))
```

## 📊 Real-world Benchmarks

Tested on standard AVX2 desktop hardware (4 heads, `dim=128`, caching up to 4096 tokens). 
*Note: We ignore sequence lengths `< 256` in these claims, as short sequences are explicitly routed to standard FP32 execution via our hybrid fallback.*

| Metric | Result (Seq ≥ 256) |
| --- | --- |
| **Latencies** | p50: `877.9 µs` \| p95: `2161.8 µs` |
| **Stable Speedup** | ~10.18x vs NumPy FP32 equivalent |
| **Throughput** | ~1,139 tokens/sec |
| **Memory** | 2.10 MB vs FP16's 8.39 MB (**4.0x smaller**) |

### Quantization Fidelity (Honest Metrics)
We use a targeted $\pm 3\sigma$ variance soft-clipping on FWHT distributions without altering Max-Lloyd codebooks. Our strictly measured empirical quality against baseline FP32:
- **Cosine Similarity**: ~0.947 (1.000 = exact identical match)
- **Mean Squared Error (MSE)**: ~1.8e-04

<img width="2400" height="600" alt="adaptq_realtime_bench" src="https://github.com/user-attachments/assets/55801a2b-f68c-4ffa-b101-a9922c94a89a" />

## 🏗 Architecture & Features

- **Unified SIMD Pipeline**: 2, 3, and 4-bit decoding share a single, quad-unrolled branchless loop using AVX2 intrinsics. No scalar fallbacks in the hot path.
- **Fast Hadamard Rotation (HAR)**: $O(d \log d)$ fully in-place rotation minimizes outliers gracefully before codebook matching.
- **Zero Heap Allocations**: Pure stack/thread-local memory buffers in the hot path.
- **Pre-Compute LUTs**: Dot products execute directly against packed indices in SIMD registers—avoiding full dequantization inside the attention kernel.
- **Plugin Architecture (V1)**: `IKVStrategy`, `IPolicy`, `IStorageBackend`, `IKernelBackend`, `ICostFunction`, `IQualityOracle` — extensible without modifying core code.
- **Replay + Compare (V2)**: Session snapshots, deterministic replay, branching at any token, and cross-strategy comparison with JSON/CSV/Markdown/LaTeX output.

---

## 🛠 Installation & Integration

### 🔹 Install from PyPI (recommended)

```bash
pip install adaptq
```

### 🔹 Install from source (latest/dev)

```bash
git clone https://github.com/l3tchupkt/adaptq.git
cd adaptq

# Build Python bindings (PyBind11)
pip install .
```

### 🔹 Build C++ CLI (for replay/compare commands)

```bash
cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel
```

---

## 🐍 Python API

### V1 — KV Cache Engine

```python
import numpy as np
from adaptq import Engine

# Engine wraps MHAContext for multi-head attention
engine = Engine(dim=128, heads=4, bits=4, capacity=2048)
k = np.random.randn(4, 128).astype('float32')  # [heads, dim]
v = np.random.randn(4, 128).astype('float32')
q = np.random.randn(4, 128).astype('float32')

engine.append(k, v)           # feed token into AdapTQ
output = engine.compute(q)    # compressed attention output
```

### V2 — Replay + Compare

The V2 Python API wraps the `adaptq replay` and `adaptq compare` CLI subcommands.

```python
from adaptq import ReplayEngine, snapshot_info

# Inspect a snapshot file
info = snapshot_info("session.aqss")
print(info)  # {'n_tokens': 256, 'n_layers': 4, 'n_heads': 8, 'dim': 128}

# Replay a session
engine = ReplayEngine()
result = engine.replay("session.aqss", collect_metrics=True)
print(result)  # ReplayResult(strategy='har_fixed', n_tokens=256, wall_time_ms=4.21)

# Branch at token 128 (warm-up then continue inference)
result = engine.replay("session.aqss", from_token=128)

# Compare strategies side-by-side
compare = engine.compare(
    "session.aqss",
    strategies=["har_fixed", "fp_passthrough"]
)
print(compare.to_markdown())
```

### V1 — PyTorch Drop-in

```python
import torch
from adaptq import AdaptQAttention

layer = AdaptQAttention(dim=128, heads=4)
out = layer(q=torch.randn(1, 4, 128),
            k=torch.randn(1, 4, 128),
            v=torch.randn(1, 4, 128))
```

---

## 🖥 CLI (V2)

The C++ `adapTQ_demo` binary exposes three subcommands:

```bash
# Replay a session snapshot
./build_release/adapTQ_demo replay session.aqss --metrics --format md

# Compare two strategies on the same snapshot
./build_release/adapTQ_demo compare session.aqss \
  --strategies har_fixed,fp_passthrough \
  --format csv --output comparison.csv

# Scaffold a new IKVStrategy implementation
./build_release/adapTQ_demo create-strategy MyKV2027
```

**Output formats:** `json` (default), `csv`, `md` (Markdown), `tex` (LaTeX).

---

## 📦 SessionSnapshot (.aqss) Format

AdapTQ V2 introduces the `.aqss` (AdapTQ Session Snapshot) binary file format:

| Section | Contents |
|---|---|
| Header | Magic `AQSS`, version, n_layers, n_heads, dim, bits, flags |
| Per-head blocks | Compressed K/V slab data, scales, format tags |
| Strategy state | IReplayHooks serialized state (optional) |
| Token log | FP32 K/V pairs for replay (optional) |

**Key properties:**
- Little-endian, self-describing, no external dependencies
- Forward-compatible: version guard on load
- Token log enables deterministic replay with any strategy

---

## 🏛 Plugin Architecture

AdapTQ V1 provides six stable plugin interfaces:

| Interface | Purpose |
|---|---|
| `IKVStrategy` | Compression + eviction capability provider |
| `IPolicy` | Session-level strategy selection |
| `IStorageBackend` | Raw slot storage (ring buffer, slab, etc.) |
| `IKernelBackend` | FWHT + kdot + vaccum kernels |
| `ICostFunction` | Memory/latency cost modelling |
| `IQualityOracle` | Reconstruction quality estimation |

Scaffold a new strategy in seconds:

```bash
./build_release/adapTQ_demo create-strategy MyKV2027
# Creates strategies/mykv2027/strategy.cpp with full template
```

---

### llama.cpp Adapter

Using `AdapTQ` as the native KV Cache replacement during computation phase over GGML.
*(Requires using `llm_build_kqv` hooks. See `/integration/llama_cpp_patch.md` for full unified patch details.)*

```cpp
#include "adapters/adapter_llamacpp.h"
LlamaCppAdaptQAdapter adapter(n_heads, head_dim, bits, capacity, seed, v_mass, hybrid_thr);
adapter.feed_kv(head, key_array, val_array, token_pos);
adapter.attention(head, query_array, out_array);
```

---

## 🧪 Tests

```bash
# C++ — 67 tests (V1: 38, V2: 29)
cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel
cd build_release && ctest --output-on-failure -j4

# Python accuracy validation (run from repo root)
python3 tests/test_accuracy.py

# Full Python validation suite
python3 tests/run_tests.py
```

## 📝 Changelog

### V2.0.0 (2026-07-22)
- ✅ RuntimeContext orchestration (full IPolicy → IKVStrategy → IStorageBackend → IKernelBackend wire-up)
- ✅ SessionSnapshot (.aqss) versioned binary format with save/load
- ✅ ReplayEngine: full replay, branch(from_token), warm-up replay, cross-strategy replay
- ✅ CLI: `adaptq replay` + `adaptq compare` with JSON/CSV/Markdown/LaTeX output
- ✅ Python: `ReplayEngine`, `ReplayResult`, `CompareResult`, `snapshot_info` in `adaptq.replay`
- ✅ 67/67 C++ tests pass
- 🐛 Fixed: `cmd_create_strategy` namespace missing from `adaptq::` 
- 🐛 Fixed: `cache_size` snapshot captured total slots instead of K/V pairs

### V1.0.0 (initial)
- Plugin architecture: IKVStrategy, IPolicy, IStorageBackend, IKernelBackend, ICostFunction, IQualityOracle
- HARFixedStrategy, FPPassthroughStrategy, UniformPolicy
- ContiguousSlabStorage, SegmentedSlabStorage
- C API, Python bindings (pybind11), PyTorch adapter
- 38/38 C++ tests pass

## 📝 License

See active repository license policies. Developed based on *AdapTQ: Adaptive Streaming Vector Quantization for Edge-Deployed Large Language Models*.
