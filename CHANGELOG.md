# Changelog

All notable changes to AdapTQ are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-07-22 — V2: Replay + Compare

### Added

#### C++ (Core)
- **`runtime/runtime_context.h/cpp`** — `RuntimeContext` orchestration class
  - Full wire-up: `IPolicy → IKVStrategy → IStorageBackend → IKernelBackend`
  - Token log (`log_tokens=true`) for FP32 K/V capture used by snapshot replay
  - `StrategyFactory` / `StorageFactory` typedefs; runtime strategy registry
  - `strategy_factory_by_name()` and `strategy_names()` registry functions
  - `make_contiguous()` public factory for `ContiguousSlabStorage`
- **`replay/session_snapshot.h/cpp`** — `SessionSnapshot` versioned binary format
  - Magic `0x41515353` ("AQSS"), version 2, little-endian self-describing header
  - Per-head K/V slab data, scales, format tags capture
  - Optional strategy state serialization via `IReplayHooks`
  - Optional token log for deterministic replay
  - `capture()`, `save()`, `load()` API
- **`replay/replay_engine.h/cpp`** — `ReplayEngine`
  - `replay()` — full deterministic replay from token 0
  - `branch(from_token)` — warm-up replay then return ready context
  - `replay_with(strategy_name)` — cross-strategy replay
  - `ReplayReport` with per-token `ComputeMetrics`
- **`cli/cmd_replay.cpp`** — `adaptq replay <snapshot.aqss> [options]`
  - `--strategy har_fixed|fp_passthrough` — override strategy
  - `--from-token N` — branch mode
  - `--metrics` — collect per-token metrics
  - `--output <file>` — file output
  - `--format json|csv|md|tex` — output format
- **`cli/cmd_compare.cpp`** — `adaptq compare <snapshot.aqss> --strategies A,B`
  - JSON/CSV/Markdown/LaTeX output
  - Aggregated per-strategy summary: wall time, avg latency, avg quality, avg bits/dim
- **`cli/cmd_create_strategy.cpp`** — `adaptq create-strategy <Name>`
  - Scaffolds full `IKVStrategy` directory under `strategies/<name>/`

#### Tests (C++)
- `tests/unit/test_runtime_context.cpp` — 10 tests for RuntimeContext
- `tests/unit/test_session_snapshot.cpp` — 8 tests for SessionSnapshot
- `tests/unit/test_replay_engine.cpp` — 9 tests for ReplayEngine

#### Python
- `adaptq/replay.py` — `ReplayEngine`, `ReplayResult`, `CompareResult`, `snapshot_info`
  - Thin subprocess wrappers around `adaptq replay`/`compare` CLI
  - `CompareResult.to_markdown()`, `best_quality()`, `fastest()`

### Changed
- `adaptq/__init__.py` — exports `ReplayEngine`, `ReplayResult`, `CompareResult`, `snapshot_info`
- `pyproject.toml` — version `0.2.0`, Python `>=3.8`, proper classifiers and extras
- `README.md` — full V2 documentation with replay/compare usage, CLI reference, snapshot format, changelog

### Fixed
- `cli/cmd_create_strategy.cpp` — missing `adaptq::` namespace wrapper caused linker error
- `replay/session_snapshot.cpp` — `cache_size` was set from `csb->size()` (total K+V slots) instead of `csb->size()/2` (K/V pairs)
- `tests/unit/test_session_snapshot.cpp` — non-copyable `RuntimeContext` returned by value; fixed to `unique_ptr`; added missing `#include <fstream>`
- `tests/unit/test_runtime_context.cpp` — inline `extern IStorageBackend *make_contiguous()` declaration inside lambda didn't resolve `adaptq::` namespace; fixed to `adaptq::make_contiguous()`

### Test Results
- **C++**: 67/67 tests pass (V1: 38, V2: 29)
- **Python**: 5/5 validation stages pass

---

## [0.1.0] — Initial Release — V1: Plugin Architecture

### Added
- **Plugin interfaces**: `IKVStrategy`, `IPolicy`, `IStorageBackend`, `IKernelBackend`, `ICostFunction`, `IQualityOracle`
- **Strategies**: `HARFixedStrategy` (4-bit HAR quantization), `FPPassthroughStrategy` (FP32 passthrough)
- **Policy**: `UniformPolicy`
- **Storage backends**: `ContiguousSlabStorage`, `SegmentedSlabStorage`
- **Quality oracle**: default ensemble oracle
- **Cost optimizer**: `ICostFunction`-based budget management
- **C API**: `adaptq_create`, `adaptq_append`, `adaptq_compute`, `adaptq_destroy`, MHA context
- **Python bindings**: pybind11 `MHAContext` → `adaptq.Engine`
- **PyTorch adapter**: `AdaptQAttention` drop-in module
- **38/38 C++ tests** pass
- **5/5 Python validation stages** pass

[0.2.0]: https://github.com/l3tchupkt/adaptq/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/l3tchupkt/adaptq/releases/tag/v0.1.0
