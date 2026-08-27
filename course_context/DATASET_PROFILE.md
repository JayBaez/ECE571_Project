# Dataset Profile

**File:** `course/Further Consolidated Data, HnL.xlsx` (≈19 MB)
All numbers below come from directly loading the workbook with
pandas/openpyxl and computing them — nothing here is assumed or guessed.

**Specified vs. Verified — read this first.** Every factual claim in
this file is now labeled as one of:
- **[SPECIFIED]** — stated in the written project spec, not
  independently re-derived here.
- **[VERIFIED]** — directly computed from the actual workbook, in this
  file's original Phase 0 pass and/or re-confirmed programmatically in
  Phase 3's `scripts/run_eda.py` (see `course_context/EDA_REPORT.md`
  for the full Phase 3 analysis this file summarizes the dataset-level
  facts from). Where a spec claim and a verified number differ, both
  are shown, and the verified number is what the code should trust.

## Sheets (9 total, 5 distinct cities) — [VERIFIED]

| Sheet name | City | Years | Rows | Notes |
|---|---|---|---|---|
| `Amhst 5hr-daily '18-'20` | Amherst | 2018–2020 | 12,056 | Only date range for Amherst |
| `Davis 5hr-daily '14-'16` | Davis | 2014–2016 | 12,056 | **Redundant** — see below |
| `Huron 5hr-daily '14-'16` | Huron | 2014–2016 | 12,056 | **Redundant** — see below |
| `Snt.Barb 5hr-daily '14-'16` | Santa Barbara | 2014–2016 | 12,056 | **Redundant** — see below |
| `LaJolla 5hr-daily '14-'16` | La Jolla | 2014–2016 | 12,056 | **Redundant** — see below |
| `Davis 5hr-daily '11-'16` | Davis | 2011–2016 | 24,112 | Superset of the '14-'16 sheet |
| `Huron 5hr-daily '11-'16` | Huron | 2011–2016 | 24,112 | Superset of the '14-'16 sheet |
| `Snt.Barb 5hr-daily '11-'16` | Santa Barbara | 2011–2016 | 24,112 | Superset of the '14-'16 sheet |
| `LaJolla 5hr-daily '11-'16` | La Jolla | 2011–2016 | 24,112 | Superset of the '14-'16 sheet |

**Update — the actual spec explains the redundant sheets, and needs both.**
For Davis, Huron, Santa Barbara, and La Jolla, the `'14-'16` sheet is
numerically identical (to floating-point rounding, max abs diff ~1e-14) to
the 2014–2016 subset of the corresponding `'11-'16` sheet — confirmed by
direct row-by-row comparison. There are really only **5 distinct
city-datasets** worth of *unique* rows in this workbook, not 9. **However,
now that the actual project spec is available (see
`TEACHER_EXPECTATIONS.md`), this redundancy is intentional, not a
data-quality problem to "fix": §3.1 explicitly requires a "3-year vs
6-year ablation" — training on a city's 3-year sheet vs. its 6-year sheet
to see whether more history helps.** This directly needs both sheets per
city. **Correction to earlier guidance in this file:** do NOT drop the
`'14-'16` sheets from the loader — keep both the short and long sheet per
city, and expose which one was used as an explicit experiment parameter.

**Important discrepancy #2 — no year overlap with Amherst.** Amherst
(2018–2020) shares zero calendar years with any of the other four cities
(2011–2016). This has a direct effect on Problem 5 (transfer learning,
Davis→Amherst per the project description): there is no way to compare
source and target on the *same* calendar dates/seasons — any cross-city or
transfer comparison is implicitly also a cross-time-period comparison, and
should be described that way rather than assumed to be climate-only.

## Columns (22, identical across all 9 sheets) — [VERIFIED]

`Year, Month, Day, Hour, Minute, DHI, DNI, GHI, Clearsky DHI, Clearsky DNI,
Clearsky GHI, Cloud Type, Dew Point, Solar Zenith Angle, Surface Albedo,
Wind Speed, Precipitable Water, Wind Direction, Relative Humidity,
Temperature, Pressure, Output Power`

This is (aside from `Output Power`) the standard NREL **NSRDB** (National
Solar Radiation Database) column set — confirmed against NREL's own API
documentation, which lists exactly this set of fields (Clearsky DHI/DNI/GHI,
Dew Point, DHI/DNI/GHI, Solar Zenith Angle, Temperature, Pressure, Relative
Humidity, Precipitable Water, Wind Direction, Wind Speed, Surface Albedo,
Cloud Type). `Output Power` is the one added column — almost certainly a
simulated/derived PV plant output, not a raw NSRDB field, since NSRDB
itself doesn't publish a power-output variable.

**Cloud Type is a categorical code, not a continuous variable.** Per NREL's
published NSRDB cloud-type coding (confirmed via NREL developer docs), the
values found in this workbook (0,1,2,3,4,6,7,8,9,10,12) map to:

| Code | Meaning |
|---|---|
| 0 | Clear |
| 1 | Probably Clear |
| 2 | Fog |
| 3 | Water |
| 4 | Super-Cooled Water |
| 6 | Opaque Ice |
| 7 | Cirrus |
| 8 | Overlapping |
| 9 | Overshooting |
| 10 | Unknown |
| 12 | Smoke |

(Codes 5 = Mixed and 11 = Dust exist in the NSRDB scheme but were not
observed in this data.) `0` (Clear) is by far the most common value in
every sheet (roughly 55–65% of rows). This column is the natural basis for
a Problem 1 "sky condition" classification target.

## Sampling structure ("5hr-daily") — [VERIFIED]

Every sheet samples **11 timestamps per calendar day**: `Hour` in
`{10, 11, 12, 13, 14, 15}` crossed with `Minute` in `{0, 30}`, minus the
`15:30` slot (i.e., 10:00, 10:30, …, 15:00). This is a fixed 5-hour daytime
window around solar noon, at 30-minute resolution — it explains the "5hr"
in the sheet names. **Every single calendar day in each city's date range
is present with exactly 11 rows — there are no missing days.** This is a
deliberate daytime-only sampling design, not a data-quality problem.

**Important discrepancy #3 — spec text vs. actual sampling window.** The
actual project spec describes the window as "between 10:00 and ~14:30
local solar time" with "~4,019 rows" per city-year. The real data instead
runs **10:00 to 15:00 inclusive** (11 samples/day, not 10 — a full 5-hour
window, matching the "5hr-daily" sheet name, not 10:00–14:30's 4.5
hours), and actual rows/year measured directly range 4,015–4,026 (close
to "~4,019," the small variation just being leap years — 2012 and 2016
have 4,026 rows, other years have 4,015). **Recommendation:** trust the
actual data over the spec's prose description for anything code depends
on (e.g., don't hard-code a 14:30 cutoff) — the "~4,019 rows"
approximation is fine to use loosely, but the literal "10:00 to ~14:30"
description undercounts the real window by about an hour.

## Missing values — [VERIFIED, re-confirmed in Phase 3]

Essentially none. The only missing data in the entire workbook is **4 rows
in the Amherst sheet**: `Output Power` is `NaN` for 2020-07-06, hours
10:00, 10:30, 11:00, 11:30 — a clear-sky day (`Cloud Type = 0`, `GHI`
exactly equal to `Clearsky GHI`) where the power reading itself is simply
missing. Worth a short imputation/drop decision when this row range is
encountered, and worth noting in the report as the one genuine missing-data
gap in the dataset.

## A real (non-obvious) data-quality anomaly: 2012-03-22 — [VERIFIED]

On **March 22, 2012**, all 11 daytime readings show `Output Power = 0`
simultaneously across **all four** of Davis, Huron, Santa Barbara, and La
Jolla (visible only in the `'11-'16` sheets, since 2012 isn't in the
`'14-'16` subset — easy to miss if you only load the shorter sheets).
Critically, this is **not a physically plausible zero**: `GHI` that day
ranges roughly 470–970 W/m² (comparable to a normal clear/partly-cloudy
day), so real solar generation should have been substantial. This looks
like a shared data-pipeline artifact (e.g., a simulation/logging outage on
that specific date, possibly a bug or shared missing-data placeholder in
whatever script generated `Output Power`) rather than four independent,
coincidentally-simultaneous real equipment outages.

**Recommendation:** treat 2012-03-22 as a known anomaly. Depending on the
experiment, either exclude it explicitly (documenting why) or keep it but
flag it in error analysis — don't let it silently count as a legitimate
"generation regime = zero output" example, since it would mislabel a
sunny day as producing no power.

Aside from this one date, no negative-irradiance values, no `GHI` values
implausibly exceeding `Clearsky GHI` (checked at a 1.2x threshold — no
violations found), and no duplicate rows were found in any sheet.

Negative values in `Temperature` and `Dew Point` (both in Celsius) are
common across all sheets in winter months — this is expected and **not** a
data-quality issue.

## A second real, non-obvious data-quality anomaly: Relative Humidity / Wind Direction swap — [VERIFIED, found in Phase 3]

`Relative Humidity` shows values above 100% (up to 360) in a subset of
rows. Investigated properly rather than dismissed as noise (full
analysis: `course_context/EDA_REPORT.md`) — the pattern is clean and
isolated:

| City | Affected rows | % of city | Affected year(s) |
|---|---|---|---|
| Davis | 3,480 | 14.4% | **2013 only** |
| Huron | 3,287 | 13.6% | **2012 only** |

In these rows, `Relative Humidity` reaches up to 360 (exactly
`Wind Direction`'s normal 0–360° range), while `Wind Direction` itself
sits under ~4.5 (consistent with radians, or some other non-degree
unit — nowhere near its normal 0–360° range in every other year).
This strongly suggests the two columns were swapped and/or recorded in
a different unit for exactly these two city-years. Not corrected here
— see `course_context/EDA_REPORT.md`'s "Decisions That Need To Be Made
Later" for the options.

## Output Power ranges by city (this matters for cross-city work) — [VERIFIED, cross-checked against SPECIFIED values]

Per the actual spec, `Output Power` is in **kW**. — [SPECIFIED]

| City | Mean (kW) | Std (kW) | Min (kW) | Max (kW) |
|---|---|---|---|---|
| Amherst | 60.9 | 41.1 | 0.0 | 132.0 |
| Davis | 164.0 | 67.4 | 0.0* | 262.8 |
| Huron | 50.1 | 16.4 | 0.0* | 77.0 |
| Santa Barbara | 49.1 | 18.3 | 0.0* | 76.7 |
| La Jolla | 47.3 | 15.6 | 0.0* | 72.3 |

(*Davis/Huron/Santa Barbara/La Jolla minimums of 0.0 in the `'11-'16`
sheets are driven by the 2012-03-22 anomaly above, not a genuine
zero-output reading — see that section.)

**Cross-check against the real spec:** the spec states max Output Power
"ranges from ~72 kW in La Jolla to ~263 kW in Davis" — this matches the
directly-measured maxima above (72.3 and 262.8) almost exactly, which is
a good independent confirmation that the dataset inspection here is
accurate.

Davis's plant clearly has roughly 3–3.5x the capacity of Huron/Santa
Barbara/La Jolla, and Amherst sits in between. **Any cross-city
comparison (Problem 2 cross-city experiment, Problem 5 transfer learning)
needs to account for this scale difference** — this is exactly why the
project spec calls for **nRMSE** (normalized RMSE) rather than raw RMSE
for cross-city regression comparisons, and it's worth normalizing/scaling
`Output Power` per city (e.g., dividing by that city's max or by installed
capacity if known) before any cross-city model training.

## Recommended defaults for the eventual data loader — [VERIFIED / implemented]

- Discover sheets by city name; map `Amhst` → `Amherst` (the loader should
  not assume the literal sheet-name spelling is the canonical city name).
  **Implemented:** `src/data_loader.py`'s `get_sheet_name()`/`load_city()`.
- Keep **both** the `'11-'16'` and `'14-'16` sheets available per city
  (not "skip the redundant ones" — see the correction above: the
  spec's §3.1 3-year-vs-6-year ablation needs both). **Implemented:**
  `load_city(city, years="long"|"short")`.
- Treat `Cloud Type` as categorical (one-hot or embedding), not ordinal/
  continuous. **Implemented:** `src/preprocessing.py`'s
  `fit_encoder()`/`apply_encoder()`.
- Keep `Year/Month/Day/Hour/Minute` available for constructing a proper
  datetime index and for chronological splitting; don't drop them as
  "just IDs."
- Be aware of, and handle explicitly (don't silently coerce): the 4
  missing `Output Power` rows in Amherst, the 2012-03-22 anomaly across
  the other four cities, and (new in Phase 3) the Davis-2013/
  Huron-2012 Relative Humidity/Wind Direction anomaly above.

## See also

- `course_context/EDA_REPORT.md` — the full Phase 3 exploratory
  analysis (distributions, correlations, class imbalance, temporal
  dependence) this file's dataset-level facts feed into.
- `course_context/LEAKAGE_MAP.md` — per-problem feature safety rules
  built from these verified facts.
