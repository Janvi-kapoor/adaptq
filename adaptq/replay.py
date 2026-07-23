"""
adaptq.replay — Python interface to AdapTQ V2 Replay + Compare

Provides Python-friendly wrappers around the C++ ReplayEngine,
SessionSnapshot, and RuntimeContext for session capture, deterministic
replay, branching, and cross-strategy comparison.

These wrappers use the subprocess CLI (adaptq binary) since the pybind11
extension currently exposes V1 MHAContext only. A native pybind11 binding
for ReplayEngine is planned for V3.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

__all__ = [
    "ReplayEngine",
    "ReplayResult",
    "CompareResult",
    "snapshot_info",
]


# ---------------------------------------------------------------------------
# Locate the adaptq CLI binary
# ---------------------------------------------------------------------------

def _find_binary() -> Optional[str]:
    """Find the adaptq CLI binary (build_v2/adapTQ_demo or system path)."""
    # 1. Check alongside this package (editable installs)
    pkg_dir = Path(__file__).parent.parent
    import sys
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    binary_name = f"adapTQ_demo{exe_suffix}"
    
    candidates = [
        pkg_dir / "build_release" / binary_name,
        pkg_dir / "build" / binary_name,
        pkg_dir / "build_v2" / binary_name,   # legacy fallback
        pkg_dir / binary_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # 2. Fall back to PATH
    import shutil
    found = shutil.which("adaptq") or shutil.which("adapTQ_demo")
    return found


_BINARY: Optional[str] = None


def _get_binary() -> str:
    global _BINARY
    if _BINARY is None:
        _BINARY = _find_binary()
    if _BINARY is None:
        raise RuntimeError(
            "adaptq CLI binary not found. "
            "Build the project first:\n"
            "  cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build_release --parallel\n"
            "Or install from source:\n"
            "  pip install ."
        )
    return _BINARY


# ---------------------------------------------------------------------------
# ReplayResult — Python-side view of ReplayReport
# ---------------------------------------------------------------------------

class ReplayResult:
    """
    Result of a replay or compare run.

    Attributes
    ----------
    strategy : str
        Name of the strategy used.
    n_tokens_replayed : int
        Number of tokens replayed.
    wall_time_ms : float
        Total wall time in milliseconds.
    metrics : list[dict]
        Per-token metrics (only populated if collect_metrics=True).
    raw : dict
        Full parsed JSON from the CLI.
    """

    def __init__(self, raw: dict):
        self.raw = raw
        self.strategy: str = raw.get("strategy", "")
        self.n_tokens_replayed: int = raw.get("n_tokens_replayed", 0)
        self.wall_time_ms: float = raw.get("wall_time_ms", 0.0)
        self.metrics: List[dict] = raw.get("metrics", [])

    def __repr__(self) -> str:
        return (
            f"ReplayResult(strategy={self.strategy!r}, "
            f"n_tokens={self.n_tokens_replayed}, "
            f"wall_time_ms={self.wall_time_ms:.2f})"
        )


class CompareResult:
    """
    Result of a multi-strategy compare run.

    Attributes
    ----------
    rows : list[dict]
        One dict per strategy with keys: strategy, n_tokens, wall_time_ms,
        avg_latency_us, avg_quality, avg_bits_per_dim.
    """

    def __init__(self, rows: List[dict]):
        self.rows = rows

    def best_quality(self) -> Optional[dict]:
        """Return the row with the highest avg_quality."""
        valid = [r for r in self.rows if r.get("avg_quality", -1) >= 0]
        if not valid:
            return None
        return max(valid, key=lambda r: r["avg_quality"])

    def fastest(self) -> Optional[dict]:
        """Return the row with the lowest wall_time_ms."""
        if not self.rows:
            return None
        return min(self.rows, key=lambda r: r.get("wall_time_ms", float("inf")))

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown table."""
        if not self.rows:
            return "*(no results)*"
        header = "| Strategy | Tokens | Wall (ms) | Latency µs | Quality | Bits/Dim |"
        sep    = "|---|---|---|---|---|---|"
        lines  = [header, sep]
        for r in self.rows:
            lines.append(
                f"| `{r.get('strategy', '?')}` "
                f"| {r.get('n_tokens', 0)} "
                f"| {r.get('wall_time_ms', 0):.2f} "
                f"| {r.get('avg_latency_us', -1):.2f} "
                f"| {r.get('avg_quality', -1):.4f} "
                f"| {r.get('avg_bits_per_dim', -1):.2f} |"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CompareResult({len(self.rows)} strategies)"


# ---------------------------------------------------------------------------
# ReplayEngine — Python interface
# ---------------------------------------------------------------------------

class ReplayEngine:
    """
    Python interface to the AdapTQ V2 ReplayEngine.

    Uses the ``adaptq replay`` and ``adaptq compare`` CLI subcommands
    internally. All I/O goes through temporary files to avoid piping
    binary snapshot data.

    Example
    -------
    >>> engine = ReplayEngine()
    >>> result = engine.replay("session.aqss", collect_metrics=True)
    >>> print(result)
    ReplayResult(strategy='har_fixed', n_tokens=128, wall_time_ms=3.14)

    >>> compare = engine.compare("session.aqss",
    ...                          strategies=["har_fixed", "fp_passthrough"])
    >>> print(compare.to_markdown())
    """

    def replay(
        self,
        snapshot_path: Union[str, Path],
        *,
        strategy: Optional[str] = None,
        from_token: Optional[int] = None,
        collect_metrics: bool = False,
        output_format: str = "json",
        output_path: Optional[Union[str, Path]] = None,
    ) -> ReplayResult:
        """
        Replay a session snapshot.

        Parameters
        ----------
        snapshot_path : str or Path
            Path to the ``.aqss`` snapshot file.
        strategy : str, optional
            Override strategy (``"har_fixed"`` or ``"fp_passthrough"``).
            If omitted, uses the strategy embedded in the snapshot config.
        from_token : int, optional
            Branch mode: warm-up to this token index then return.
        collect_metrics : bool
            Collect per-token ComputeMetrics (slower but richer output).
        output_format : str
            Output format: ``"json"`` (default), ``"csv"``, ``"md"``, ``"tex"``.
        output_path : str or Path, optional
            Write output to this file instead of capturing it.

        Returns
        -------
        ReplayResult
        """
        binary = _get_binary()
        cmd = [binary, "replay", str(snapshot_path)]
        if strategy:
            cmd += ["--strategy", strategy]
        if from_token is not None:
            cmd += ["--from-token", str(from_token)]
        if collect_metrics:
            cmd.append("--metrics")
        # Note: output_format used only if output_path specified; internal capture always uses json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Always capture JSON internally; write to user path separately if requested
            cmd += ["--format", "json", "--output", tmp_path]
            result = subprocess.run(
                cmd, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"adaptq replay failed:\n{result.stderr}"
                )
            with open(tmp_path) as f:
                raw = json.load(f)
            return ReplayResult(raw)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def compare(
        self,
        snapshot_path: Union[str, Path],
        *,
        strategies: List[str],
        output_format: str = "json",
        output_path: Optional[Union[str, Path]] = None,
    ) -> CompareResult:
        """
        Compare multiple strategies on the same session snapshot.

        Parameters
        ----------
        snapshot_path : str or Path
            Path to the ``.aqss`` snapshot file.
        strategies : list[str]
            Strategy names to compare (e.g. ``["har_fixed", "fp_passthrough"]``).
        output_format : str
            Output format for file output (``"json"``, ``"csv"``, ``"md"``, ``"tex"``).
        output_path : str or Path, optional
            Write comparison output to file (optional).

        Returns
        -------
        CompareResult
        """
        if not strategies:
            raise ValueError("strategies list must be non-empty")

        binary = _get_binary()
        strats_csv = ",".join(strategies)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                binary, "compare", str(snapshot_path),
                "--strategies", strats_csv,
                "--format", "json",
                "--output", tmp_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"adaptq compare failed:\n{result.stderr}"
                )
            with open(tmp_path) as f:
                rows = json.load(f)
            return CompareResult(rows)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Snapshot utilities
# ---------------------------------------------------------------------------

def snapshot_info(path: Union[str, Path]) -> dict:
    """
    Return basic metadata from a ``.aqss`` snapshot file without full replay.

    This loads the snapshot header only by running ``adaptq replay`` with
    0 tokens and JSON output.

    Returns
    -------
    dict with keys: n_tokens, n_layers, n_heads, dim, has_token_log,
    has_strategy_state.
    """
    binary = _get_binary()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [binary, "replay", str(path),
               "--from-token", "0",
               "--format", "json",
               "--output", tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"adaptq replay failed:\n{result.stderr}")
        with open(tmp_path) as f:
            raw = json.load(f)
        return {
            "n_tokens": raw.get("snapshot_n_tokens", 0),
            "n_layers": raw.get("n_layers", 0),
            "n_heads":  raw.get("n_heads", 0),
            "dim":      raw.get("dim", 0),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
