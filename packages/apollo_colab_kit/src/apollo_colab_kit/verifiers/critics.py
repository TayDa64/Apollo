"""Pure-Python deterministic quality-gate critics (VLC, BKAS, ADC).

Mirrors ``multimodal_gen.intelligence.critics`` with zero heavy dependencies.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class CriticResult:
    """Result of a single quality-gate metric."""

    name: str
    value: float
    threshold: float
    passed: bool
    details: str


@dataclass
class CriticReport:
    """Aggregated report from all critics."""

    metrics: List[CriticResult]
    overall_passed: bool
    summary: str


def compute_vlc(voicings: Sequence[Sequence[int]]) -> CriticResult:
    """Voice Leading Cost — average semitone movement per chord change (threshold < 3.0)."""
    threshold = 3.0

    if len(voicings) < 2:
        return CriticResult(
            name="VLC",
            value=0.0,
            threshold=threshold,
            passed=True,
            details="Fewer than 2 voicings — nothing to measure.",
        )

    non_empty = [list(v) for v in voicings if v]
    if len(non_empty) < 2:
        return CriticResult(
            name="VLC",
            value=0.0,
            threshold=threshold,
            passed=True,
            details="Fewer than 2 non-empty voicings — nothing to measure.",
        )

    costs: List[float] = []
    for i in range(1, len(non_empty)):
        a = sorted(non_empty[i - 1])
        b = sorted(non_empty[i])

        while len(a) < len(b):
            a.append(a[-1] if a else 60)
        while len(b) < len(a):
            b.append(b[-1] if b else 60)

        cost = sum(abs(x - y) for x, y in zip(a, b)) / max(len(a), 1)
        costs.append(cost)

    avg_cost = statistics.mean(costs)
    passed = avg_cost < threshold

    details = (
        f"Avg voice-leading cost: {avg_cost:.2f} semitones/voice over "
        f"{len(costs)} transition(s). "
        f"{'Smooth voice leading.' if passed else 'Large jumps detected — consider closer voicings.'}"
    )

    return CriticResult(
        name="VLC",
        value=round(avg_cost, 4),
        threshold=threshold,
        passed=passed,
        details=details,
    )


def compute_bkas(
    bass_notes: Sequence[Dict[str, int]],
    kick_ticks: Sequence[int],
    tolerance_ticks: int = 60,
) -> CriticResult:
    """Bass-Kick Alignment Score — fraction of kick hits near a bass note (threshold >= 0.70)."""
    threshold = 0.70

    if not kick_ticks:
        return CriticResult(
            name="BKAS",
            value=1.0,
            threshold=threshold,
            passed=True,
            details="No kick events — nothing to measure.",
        )

    bass_ticks = sorted(bn.get("tick", bn.get("start_tick", 0)) for bn in bass_notes)

    aligned = 0
    for kt in kick_ticks:
        for bt in bass_ticks:
            if abs(kt - bt) <= tolerance_ticks:
                aligned += 1
                break

    score = aligned / len(kick_ticks)
    passed = score >= threshold

    details = (
        f"{aligned}/{len(kick_ticks)} kick hits aligned with bass "
        f"(tolerance ±{tolerance_ticks} ticks). "
        f"Score: {score:.1%}. "
        f"{'Good bass-kick lock.' if passed else 'Bass and kick are misaligned — tighten pocket.'}"
    )

    return CriticResult(
        name="BKAS",
        value=round(score, 4),
        threshold=threshold,
        passed=passed,
        details=details,
    )


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 on degenerate input."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xs_slice, ys_slice = list(xs[:n]), list(ys[:n])
    mean_x = statistics.mean(xs_slice)
    mean_y = statistics.mean(ys_slice)

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_slice, ys_slice))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs_slice))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys_slice))

    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def compute_adc(
    tension_curve: Sequence[float],
    note_density_per_beat: Sequence[float],
) -> CriticResult:
    """Arrangement Density Curve — Pearson correlation between tension and density (threshold >= 0.6)."""
    threshold = 0.6

    n = min(len(tension_curve), len(note_density_per_beat))
    if n < 2:
        return CriticResult(
            name="ADC",
            value=0.0,
            threshold=threshold,
            passed=False,
            details="Not enough data points to compute correlation.",
        )

    r = _pearson_correlation(tension_curve[:n], note_density_per_beat[:n])
    passed = r >= threshold

    details = (
        f"Pearson r between tension and density over {n} time-slices: "
        f"{r:.3f}. "
        f"{'Density follows tension well.' if passed else 'Density is decoupled from tension — revise arrangement.'}"
    )

    return CriticResult(
        name="ADC",
        value=round(r, 4),
        threshold=threshold,
        passed=passed,
        details=details,
    )


def run_all_critics(
    voicings: Optional[Sequence[Sequence[int]]] = None,
    bass_notes: Optional[Sequence[Dict[str, int]]] = None,
    kick_ticks: Optional[Sequence[int]] = None,
    tolerance_ticks: int = 60,
    tension_curve: Optional[Sequence[float]] = None,
    note_density_per_beat: Optional[Sequence[float]] = None,
) -> CriticReport:
    """Run VLC, BKAS, and ADC critics and return a unified ``CriticReport``."""
    metrics: List[CriticResult] = [
        compute_vlc(voicings or []),
        compute_bkas(bass_notes or [], kick_ticks or [], tolerance_ticks),
        compute_adc(tension_curve or [], note_density_per_beat or []),
    ]

    overall = all(m.passed for m in metrics)
    passed_count = sum(1 for m in metrics if m.passed)
    total = len(metrics)

    lines = [f"  {m.name}: {'PASS' if m.passed else 'FAIL'} ({m.value:.3f})" for m in metrics]
    summary = (
        f"Quality gate: {passed_count}/{total} critics passed"
        f" — {'ALL PASS' if overall else 'GATE FAILED'}.\n"
        + "\n".join(lines)
    )

    logger.info("Critics report: %d/%d passed", passed_count, total)
    return CriticReport(metrics=metrics, overall_passed=overall, summary=summary)
