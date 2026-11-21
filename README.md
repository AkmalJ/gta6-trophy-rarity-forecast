# GTA VI Trophy Rarity Forecaster

Entry for make.fun's **gta6-trophy-rarity-forecast-01**: forecast every
PlayStation 5 GTA VI trophy's Exophase earn rate at the case target.

`main.py` (Python 3.13, stdlib only, no network) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `/output/<id>.out` (dir from
`$MAKEFUN_OUTPUT_DIR` else `/output`) with five quantiles per trophy at the
levels 0.05, 0.25, 0.5, 0.75, 0.95.

Method, all on the logit scale where the pinball loss is scored:

- Per trophy, build the series of distinct rates (consecutive duplicates are
  Exophase refresh steps and collapse, keeping the last time seen).
- Fit a least-squares logit trend over the last up-to-12 distinct points and
  project it with a saturating horizon, `H / (1 + H/72)`, so week-long
  projections bend toward an asymptote instead of exploding.
- Cap the median by a per-trophy ceiling (`min(99.5, max(3x last, last+25))`,
  tighter for the platinum); a trophy already at 100 stays pinned near 100;
  a trophy with no history borrows its tier's median last rate.
- Spread from the fit residuals: `s = 0.25 + 0.9 * max(resid_std, 0.15) *
  sqrt(1 + H/24)`, quantiles at the normal z values for the five levels.

Deterministic: the same case always produces the same bytes.
