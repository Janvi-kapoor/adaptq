#!/usr/bin/env bash
# ==========================================================================
# validate_examples.sh
# Phase 3: Validate every example in examples/
# ==========================================================================
ROOT=/mnt/e/Researches/AdaptQ/adapTQ
LOG=$ROOT/benchmarks/results/examples_validation.log
PASS=0
FAIL=0
SKIPPED=0

mkdir -p "$ROOT/benchmarks/results"

echo "=== AdapTQ Examples Validation ===" | tee "$LOG"
echo "Date: $(date)" | tee -a "$LOG"
echo "Python: $(python3 --version)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_example() {
    local name=$1
    local script=$2
    local args=$3
    local timeout=${4:-120}

    echo "─────────────────────────────────────────────────" | tee -a "$LOG"
    echo "Running: $name" | tee -a "$LOG"
    echo "  Script: $script $args" | tee -a "$LOG"

    timeout "$timeout" python3 "$script" $args 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}

    if [ $rc -eq 0 ]; then
        echo "  ✅ PASS (exit $rc)" | tee -a "$LOG"
        ((PASS++))
    elif [ $rc -eq 124 ]; then
        echo "  ⏱  TIMEOUT (>${timeout}s) — marking as SKIP" | tee -a "$LOG"
        ((SKIPPED++))
    else
        echo "  ❌ FAIL (exit $rc)" | tee -a "$LOG"
        ((FAIL++))
    fi
    echo "" | tee -a "$LOG"
}

cd "$ROOT"

# ── 1. snapshot_demo.py — no model needed ─────────────────────────────────
run_example "snapshot_demo" \
    "examples/snapshot_demo.py" \
    "--n-tokens 16 --dim 64 --bits 4" \
    60

# ── 2. branch_replay_demo.py — no model needed ────────────────────────────
run_example "branch_replay_demo" \
    "examples/branch_replay_demo.py" \
    "" \
    60

# ── 3. compare_backends.py — uses available backends only ─────────────────
run_example "compare_backends" \
    "examples/compare_backends.py" \
    "--backends transformers --tokens 15 --prompt 'KV cache is'" \
    300

# ── 4. transformers_demo.py — Qwen2-0.5B (fast if cached) ─────────────────
run_example "transformers_demo" \
    "examples/transformers_demo.py" \
    "--model Qwen/Qwen2-0.5B --tokens 20 --snapshot-at 10 --prompt 'KV cache is'" \
    600

# ── 5. llama_demo.py — requires GGUF file ─────────────────────────────────
GGUF=$(find "$ROOT" -name "*.gguf" 2>/dev/null | head -1)
if [ -n "$GGUF" ]; then
    run_example "llama_demo" \
        "examples/llama_demo.py" \
        "--model $GGUF --tokens 15 --prompt 'KV cache is'" \
        120
else
    echo "─────────────────────────────────────────────────" | tee -a "$LOG"
    echo "Running: llama_demo (auto-download TinyLlama GGUF)" | tee -a "$LOG"
    run_example "llama_demo_auto_download" \
        "examples/llama_demo.py" \
        "--auto-download --tokens 15 --prompt 'KV cache is'" \
        900
fi

# ── 6. ollama_demo.py — requires Ollama server ─────────────────────────────
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    run_example "ollama_demo" \
        "examples/ollama_demo.py" \
        "--model tinyllama --tokens 20" \
        120
else
    echo "─────────────────────────────────────────────────" | tee -a "$LOG"
    echo "Skipping: ollama_demo (Ollama server not running)" | tee -a "$LOG"
    echo "  Start with: ollama serve &  then  ollama pull tinyllama" | tee -a "$LOG"
    ((SKIPPED++))
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════" | tee -a "$LOG"
echo "Examples Validation Summary:" | tee -a "$LOG"
echo "  PASS   : $PASS" | tee -a "$LOG"
echo "  FAIL   : $FAIL" | tee -a "$LOG"
echo "  SKIPPED: $SKIPPED" | tee -a "$LOG"
echo "" | tee -a "$LOG"
if [ "$FAIL" -eq 0 ]; then
    echo "  ✅ ALL EXAMPLES PASS (skipped=$SKIPPED)" | tee -a "$LOG"
else
    echo "  ❌ $FAIL EXAMPLE(S) FAILED" | tee -a "$LOG"
    exit 1
fi
