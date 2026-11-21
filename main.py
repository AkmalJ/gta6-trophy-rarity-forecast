# SPDX-License-Identifier: MIT
"""GTA VI trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes
/output/<id>.out with five quantiles per trophy. The whole model lives on
the logit scale, where the pinball loss is scored: each trophy's recent
logit trend is projected forward with a saturating horizon, capped by a
per-trophy ceiling, and given an honest spread from the fit residuals.
Stdlib only, no network, deterministic.
"""

import glob
import json
import math
import os
from datetime import datetime

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)  # normal quantiles at the 5 levels
P_LO, P_HI = 0.001, 99.999               # percent clamp before the logit


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, P_LO, P_HI)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def series_for(history, idx):
    """(hours_since_first_read, rate) points for one trophy.

    Only reads where the rate is present count. Consecutive duplicate
    rates are Exophase refresh steps, not new information: collapse them,
    keeping the LAST time each distinct value was seen.
    """
    pts = []
    t0 = None
    for read in history:
        rate = read["rates"][idx]
        if rate is None:
            continue
        t = parse_time(read["at"])
        if t0 is None:
            t0 = t
        th = (t - t0) / 3600.0
        if pts and pts[-1][1] == rate:
            pts[-1] = (th, rate)
        else:
            pts.append((th, rate))
    return pts


def fit_trend(pts):
    """Least-squares slope on logit over the last <=12 distinct points.

    Returns (last_logit, slope_per_hour, residual_std). With fewer than
    two distinct points there is no trend: slope 0 and a default spread.
    """
    last_logit = logit(pts[-1][1])
    if len(pts) < 2:
        return last_logit, 0.0, 0.30
    use = pts[-12:]
    xs = [p[0] for p in use]
    ys = [logit(p[1]) for p in use]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return last_logit, 0.0, 0.30
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = clamp(sxy / sxx, -0.5, 0.5)
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return last_logit, slope, math.sqrt(var)


def tier_medians(case):
    """Median last observed rate per tier, for trophies with no history."""
    by_tier = {}
    for i, tr in enumerate(case["trophies"]):
        pts = series_for(case["history"], i)
        if pts:
            by_tier.setdefault(tr["tier"], []).append(pts[-1][1])
    out = {}
    for tier, vals in by_tier.items():
        vals.sort()
        out[tier] = vals[len(vals) // 2]
    return out


def forecast_row(case, idx, horizon_h, tier_med):
    """Five quantiles (percent) for one trophy."""
    tr = case["trophies"][idx]
    pts = series_for(case["history"], idx)
    if not pts:
        # No reads at all: borrow the tier's typical level, wide spread.
        med = tier_med.get(tr["tier"], 1.0)
        med_logit = logit(med)
        s = 1.2 * math.sqrt(1.0 + horizon_h / 24.0)
        return quantiles(med_logit, s)
    last = pts[-1][1]
    if last >= 100.0:
        # Already universal: stay pinned just under 100, narrow spread.
        return quantiles(logit(99.95), 0.05)
    last_logit, slope, resid_std = fit_trend(pts)
    # Saturating horizon: long projections bend toward an asymptote
    # instead of running the logit trend off to 0 or 100.
    h_eff = horizon_h / (1.0 + horizon_h / 72.0)
    med_logit = last_logit + slope * h_eff
    med = inv_logit(med_logit)
    # Ceiling: a week-one rate cannot outrun its own level by much.
    cap = min(99.5, max(last * 3.0, last + 25.0))
    if tr["id"] == 1:  # the platinum stays very low in week one
        cap = min(cap, max(last * 1.5, last + 2.0))
    if last <= cap and med > cap:
        med_logit = logit(cap)
    s = 0.25 + 0.9 * max(resid_std, 0.15) * math.sqrt(1.0 + horizon_h / 24.0)
    return quantiles(med_logit, s)


def quantiles(med_logit, s):
    qs = []
    for z in Z:
        q = inv_logit(med_logit + z * s)
        qs.append(round(clamp(q, P_LO, P_HI), 4))
    for i in range(1, len(qs)):  # rounding must not break monotonicity
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    horizon_h = case["horizon_minutes"] / 60.0
    tier_med = tier_medians(case)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        rows.append({"id": tr["id"], "q": forecast_row(case, i, horizon_h, tier_med)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
