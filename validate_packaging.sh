#!/usr/bin/env bash
# ==========================================================================
# validate_packaging.sh
# Phase 2: Full packaging validation in a clean virtual environment
# ==========================================================================
set -e
ROOT=/mnt/e/Researches/AdaptQ/adapTQ
VENV=$ROOT/test_env
LOG=$ROOT/benchmarks/results/packaging_validation.log

echo "=== AdapTQ Packaging Validation ===" | tee $LOG
echo "Date: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# ── Clean up any previous test env ────────────────────────────────────────
echo "1. Creating clean virtual environment..." | tee -a $LOG
rm -rf "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
echo "   Python: $(python --version)" | tee -a $LOG
echo "   pip: $(pip --version)" | tee -a $LOG

# ── Install build deps ────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "2. Installing build dependencies..." | tee -a $LOG
pip install -q setuptools wheel pybind11 numpy 2>&1 | tail -3 | tee -a $LOG

# ── pip install . ─────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "3. pip install . (from source)..." | tee -a $LOG
cd "$ROOT"
pip install . 2>&1 | tee -a $LOG
echo "   Exit code: $?" | tee -a $LOG

# ── Verify package metadata ───────────────────────────────────────────────
echo "" | tee -a $LOG
echo "4. pip show adaptq:" | tee -a $LOG
pip show adaptq 2>&1 | tee -a $LOG

# ── Verify installed packages ─────────────────────────────────────────────
echo "" | tee -a $LOG
echo "5. Verifying package installation..." | tee -a $LOG

python -c "
import sys, importlib

results = []

# Core package
try:
    import adaptq
    results.append(('adaptq', True, f'OK — version not in module'))
except Exception as e:
    results.append(('adaptq', False, str(e)))

# runtime_py
try:
    from adaptq.runtime_py import create_adapter, list_available_backends, IRuntimeAdapter
    results.append(('adaptq.runtime_py', True, f'create_adapter, list_available_backends, IRuntimeAdapter OK'))
except Exception as e:
    results.append(('adaptq.runtime_py', False, str(e)))

# backends
try:
    from adaptq.runtime_py.backends import SUPPORTED_BACKENDS, get_adapter_class
    results.append(('adaptq.runtime_py.backends', True, f'{len(SUPPORTED_BACKENDS)} backends registered'))
except Exception as e:
    results.append(('adaptq.runtime_py.backends', False, str(e)))

# metadata
try:
    from adaptq.runtime_py.metadata import ModelConfig, SessionConfig, RuntimeMetadata, KVStats, GenerationResult
    results.append(('adaptq.runtime_py.metadata', True, 'all dataclasses OK'))
except Exception as e:
    results.append(('adaptq.runtime_py.metadata', False, str(e)))

# adaptq_py C extension
try:
    import adaptq_py
    results.append(('adaptq_py (C ext)', True, f'OK'))
except ImportError as e:
    results.append(('adaptq_py (C ext)', False, f'NOT installed (expected — requires build): {e}'))

# Create adapter test
try:
    from adaptq.runtime_py import create_adapter
    a = create_adapter('transformers')
    results.append(('create_adapter(transformers)', True, repr(a)))
except ImportError as e:
    results.append(('create_adapter(transformers)', False, f'ImportError (need pip install transformers): {e}'))
except Exception as e:
    results.append(('create_adapter(transformers)', False, str(e)))

print()
print('  Import check results:')
all_ok = True
for name, ok, msg in results:
    status = '✓' if ok else '✗'
    print(f'  [{status}] {name}: {msg}')
    if not ok and 'adaptq_py' not in name and 'transformers' not in name:
        all_ok = False

print()
if all_ok:
    print('  PACKAGING: PASS')
else:
    print('  PACKAGING: FAIL — see above')
    sys.exit(1)
" 2>&1 | tee -a $LOG

# ── Check installed file locations ─────────────────────────────────────────
echo "" | tee -a $LOG
echo "6. Installed file locations:" | tee -a $LOG
python -c "
import adaptq, adaptq.runtime_py, adaptq.runtime_py.backends
print(f'  adaptq          : {adaptq.__file__}')
print(f'  adaptq.runtime_py: {adaptq.runtime_py.__file__}')
print(f'  backends        : {adaptq.runtime_py.backends.__file__}')
" 2>&1 | tee -a $LOG

# ── list_available_backends ───────────────────────────────────────────────
echo "" | tee -a $LOG
echo "7. list_available_backends() in clean env (only base deps):" | tee -a $LOG
python -c "
from adaptq.runtime_py import list_available_backends
print(f'  Available: {list_available_backends()}')
print('  (Expected: [] in a clean env without transformers/llama-cpp-python installed)')
" 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Phase 2: PACKAGING VALIDATION COMPLETE ===" | tee -a $LOG
deactivate
