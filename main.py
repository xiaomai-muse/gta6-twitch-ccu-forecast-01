"""GTA VI launch-day Twitch concurrent viewer forecast (makefun/twitch-ccu-case/v1).

Runtime-fitted empirical log-ratio quantile model over filtered rolling-midnight
comparable launches, using only the Python standard library.
"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
FALLBACK_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
FALLBACK_SIGMA = 0.15
DEFAULT_SPREAD_MULT = 1.00
DEFAULT_H1_BANDWIDTH = 1.5
DEFAULT_SMOOTH_ALPHA = 0.25
MAX_VIEWERS = 1_000_000_000.0
MIN_VIEWERS = 0.01


def round2(x: float) -> float | int:
    """Round x to 2 decimals, half up, returning int when exact."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def _parse_iso_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def extract_comparable_hourly(comp: dict) -> dict[int, float]:
    """Map a comparable's hourly rows to release-day hours 1..24."""
    day_start_str = comp.get("day_start")
    hourly = comp.get("hourly")
    if not day_start_str or not hourly:
        return {}
    ds = _parse_iso_utc(day_start_str)
    by_hour: dict[int, float] = {}
    for row in hourly:
        ts = row.get("from")
        viewers = row.get("viewers")
        if ts is None or viewers is None:
            continue
        dt = _parse_iso_utc(ts)
        h = round((dt - ds).total_seconds() / 3600.0) + 1
        if 1 <= h <= 24:
            by_hour[h] = float(viewers)
    return by_hour


def is_comparable_eligible(
    comp: dict,
    by_hour: dict[int, float],
    min_h1_peak_ratio: float = 0.02,
    exclude_early_access: bool = True,
    exclude_weak: bool = True,
) -> bool:
    """Return True if comparable resembles a rolling-midnight paid launch."""
    if 1 not in by_hour or len(by_hour) < 12:
        return False
    unlock = (comp.get("unlock") or "").lower()
    if "not a launch" in unlock or "target-history" in unlock:
        return False
    if exclude_early_access and "early-access-contaminated" in unlock:
        return False
    if exclude_weak and (
        "tier: weak" in unlock
        or "free-to-play" in unlock
        or "weak analogue" in unlock
        or "weak gta analogue" in unlock
        or "multiplayer-f2p" in unlock
    ):
        return False
    peak = max(by_hour.values())
    if peak <= 0:
        return False
    h1 = max(1.0, by_hour[1])
    return (h1 / peak) >= min_h1_peak_ratio


def select_comparables(
    comparables: list[dict],
    min_h1_peak_ratio: float = 0.02,
    min_count: int = 10,
) -> list[dict[int, float]]:
    """Filter comparables with graceful fallback if fewer than min_count pass."""
    parsed = [(c, extract_comparable_hourly(c)) for c in comparables]

    tiers = [
        (min_h1_peak_ratio, True, True),
        (0.005, True, True),
        (0.005, True, False),
        (0.0, False, False),
    ]
    for ratio, ex_ea, ex_wk in tiers:
        kept = [
            by_h
            for c, by_h in parsed
            if is_comparable_eligible(
                c,
                by_h,
                min_h1_peak_ratio=ratio,
                exclude_early_access=ex_ea,
                exclude_weak=ex_wk,
            )
        ]
        if len(kept) >= min_count:
            return kept
    return [by_h for _, by_h in parsed if 1 in by_h and len(by_h) >= 2]


def linear_quantile(sorted_vals: list[float], tau: float) -> float:
    """Empirical quantile using linear interpolation on sorted values."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("Cannot compute quantile of empty sequence")
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def weighted_quantile(vals_weights: list[tuple[float, float]], tau: float) -> float:
    """Empirical quantile using midpoint CDF interpolation on weighted values."""
    if not vals_weights:
        raise ValueError("Cannot compute quantile of empty sequence")
    if len(vals_weights) == 1:
        return vals_weights[0][0]
    vw = sorted(vals_weights, key=lambda x: x[0])
    total = sum(w for _, w in vw)
    if total <= 0.0:
        return linear_quantile([v for v, _ in vw], tau)
    cum = 0.0
    pts: list[tuple[float, float]] = []
    for v, w in vw:
        mid = (cum + 0.5 * w) / total
        pts.append((mid, v))
        cum += w
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        p0, v0 = pts[i]
        p1, v1 = pts[i + 1]
        if p0 <= tau <= p1:
            if p1 == p0:
                return v0
            t = (tau - p0) / (p1 - p0)
            return v0 * (1.0 - t) + v1 * t
    return pts[-1][1]


def fit_log_ratio_model(
    comparables: list[dict],
    target_p1: float | None = None,
    spread_mult: float = DEFAULT_SPREAD_MULT,
    horizon_slope: float = 0.0,
    min_h1_peak_ratio: float = 0.02,
    h1_bandwidth: float = DEFAULT_H1_BANDWIDTH,
    smooth_alpha: float = DEFAULT_SMOOTH_ALPHA,
) -> dict[int, list[float]]:
    """Fit per-hour log-ratio quantiles ln(viewers_h / viewers_1) from comparables."""
    kept = select_comparables(comparables, min_h1_peak_ratio=min_h1_peak_ratio)
    if not kept:
        return {}

    log_h1s = [math.log(max(1.0, by_h[1])) for by_h in kept]
    if target_p1 is not None and h1_bandwidth > 0.0:
        lo_lh1, hi_lh1 = min(log_h1s), max(log_h1s)
        clamped_log_p1 = max(lo_lh1, min(hi_lh1, math.log(max(1.0, target_p1))))
        weights = [
            math.exp(-0.5 * ((clamped_log_p1 - lh1) / h1_bandwidth) ** 2)
            for lh1 in log_h1s
        ]
    else:
        weights = [1.0] * len(kept)

    model: dict[int, list[float]] = {}
    for h in range(2, 25):
        vw: list[tuple[float, float]] = []
        for by_h, w in zip(kept, weights):
            if h not in by_h:
                continue
            v1 = max(1.0, by_h[1])
            vh = max(1.0, by_h[h])
            vw.append((math.log(vh / v1), w))
        if not vw:
            continue
        raw_q = [weighted_quantile(vw, tau) for tau in LEVELS]
        med = raw_q[2]
        mult = spread_mult * (1.0 + horizon_slope * math.sqrt(h - 1))
        cal_q = [med + mult * (q - med) for q in raw_q]
        model[h] = cal_q

    if smooth_alpha > 0.0 and model:
        smoothed: dict[int, list[float]] = {}
        center = 1.0 - 2.0 * smooth_alpha
        for h, qs in model.items():
            if (h - 1) in model and (h + 1) in model:
                prev_q = model[h - 1]
                next_q = model[h + 1]
                smoothed[h] = [
                    smooth_alpha * prev_q[i] + center * qs[i] + smooth_alpha * next_q[i]
                    for i in range(len(qs))
                ]
            else:
                smoothed[h] = list(qs)
        return smoothed
    return model


def _sanitize_row(values: list[float]) -> list[float | int]:
    """Clamp to (0, 1e9], enforce non-decreasing, and round to 2 decimals."""
    clamped = []
    for v in values:
        if not math.isfinite(v):
            v = 1.0
        v = max(MIN_VIEWERS, min(MAX_VIEWERS, v))
        clamped.append(round2(v))
    clamped.sort()
    for i in range(len(clamped)):
        if clamped[i] <= 0:
            clamped[i] = MIN_VIEWERS
        elif clamped[i] > MAX_VIEWERS:
            clamped[i] = int(MAX_VIEWERS)
    return clamped


def _fallback_row(p1: float, h: int) -> list[float | int]:
    vals = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in FALLBACK_Z]
    return _sanitize_row(vals)


def forecast_case(
    case: dict,
    name: str,
    spread_mult: float = DEFAULT_SPREAD_MULT,
    horizon_slope: float = 0.0,
    h1_bandwidth: float = DEFAULT_H1_BANDWIDTH,
) -> dict:
    """Produce a valid forecast dict for a single case."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in (case.get("known") or []) if k.get("hour") == 1), None)
    if first is None or first.get("viewers") is None:
        p1 = 1.0
    else:
        p1 = float(max(1.0, float(first["viewers"])))

    model = fit_log_ratio_model(
        case.get("comparables") or [],
        target_p1=p1,
        spread_mult=spread_mult,
        horizon_slope=horizon_slope,
        h1_bandwidth=h1_bandwidth,
    )

    raw_forecast = case.get("forecast") or []
    forecast_hours = [
        int(r["hour"])
        for r in raw_forecast
        if isinstance(r, dict) and r.get("hour") is not None
    ]
    if sorted(forecast_hours) != list(range(2, 25)):
        forecast_hours = list(range(2, 25))

    rows = []
    for h in forecast_hours:
        if h in model:
            vals = [p1 * math.exp(max(-30.0, min(30.0, lq))) for lq in model[h]]
            q = _sanitize_row(vals)
        else:
            q = _fallback_row(p1, h)
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = {
                "hours": [
                    {"hour": h, "q": _fallback_row(1.0, h)} for h in range(2, 25)
                ]
            }
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
