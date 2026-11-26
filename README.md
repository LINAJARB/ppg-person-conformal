# Person-based conformal calibration of wrist PPG heart rate intervals

Code for the paper *Person-Based Conformal Calibration of Heart Rate Prediction Intervals from Wrist Photoplethysmography Across Wearers and Activities* (L. Jarboui).

The code estimates the heart rate (HR) from wrist PPG with MAST, a motion-attenuated spectral tracker, computes three signal descriptors, fits an error scale, and calibrates prediction intervals with the person as the calibration unit. The calibration only needs an HR estimate and a scale of its error, so it can be placed after any HR estimator. The scripts reproduce every table, figure and number of the paper.

## Data

The four datasets are public and must be downloaded from their original sources:

| Dataset | Source | Expected folder |
|---|---|---|
| PPG-DaLiA (Reiss et al., 2019) | https://doi.org/10.24432/C53890 | `PPG_FieldStudy` |
| WESAD (Schmidt et al., 2018) | https://doi.org/10.24432/C57K5T | `WESAD` |
| GalaxyPPG (Park et al., 2025) | data descriptor https://doi.org/10.1038/s41597-025-05152-z | `GalaxyPPG/Dataset` |
| PPG-Sport (Hu et al., 2026) | https://github.com/LaserHu/PPG-Sport | `PPG-Sport-main` (`dataset/`, `cleaned_gt/`) |

PPG-DaLiA, WESAD and GalaxyPPG are used through their Empatica E4 signals, with the reference HR of PPG-DaLiA and the chest ECG of WESAD and GalaxyPPG. PPG-Sport (Empatica EmbracePlus) is used with the reference HR distributed with the dataset.

## Installation

```
python -m pip install -r requirements.txt
```

The experiment with BeliefPPG needs a separate environment (Python 3.11):

```
python3.11 -m venv belief_env
belief_env/bin/python -m pip install -r beliefppg/requirements.txt
```

## Reproducing the results

Run the steps from the repository root, in this order. Each step writes its outputs in `results/`.

| Step | Script | What it does |
|---|---|---|
| 1a | `step01a_extract_compact.py <data>` | Compact copy of PPG-DaLiA and WESAD (wrist signals, references) |
| 1b | `step01b_development_windows.py` | HR estimate and descriptors for the 30 development participants, ECG quality control |
| 2 | `step02_development_calibration.py` | Leave-one-subject-out calibration on the development data, three score functions |
| 3 | `step03_galaxyppg_external.py <GalaxyPPG/Dataset>` | GalaxyPPG windows, chest clock alignment, calibration on the development data |
| 4a | `belief_env/bin/python beliefppg/run_beliefppg.py compact <GalaxyPPG/Dataset>` | BeliefPPG estimates and posterior SD |
| 4b | `step04_beliefppg_calibration.py` | BeliefPPG calibrated within GalaxyPPG |
| 5 | `step05_person_quantile.py` | Person-quantile calibration, split calibration, fold sensitivity |
| 6 | `step06_person_coverage.py` | Coverage of each participant with the three calibrations |
| 7 | `step07_ablation.py` | Ablation of the weighting and of the finite-sample correction |
| 8 | `step08_cqr_crossing.py` | Crossing of the CQR quantiles |
| 9 | `step09_calibration_rules.py` | Methods of Dunn et al., motion-stratified calibration, coverage-width curve, bootstrap, rejection |
| 10 | `step10_motion_stratified.py` | Bootstrap per activity, width per motion class, motion-stratified BeliefPPG |
| 11 | `step11_validate_rpeaks.py <data>` | Validation of the R-peak detector on PPG-DaLiA |
| 12 | `step12_figures.py` | Figures (coverage per person, coverage per condition, coverage-width curve, example recording) |
| 12b | `step12b_tradeoff_table.py` | Coverage-width trade-off against one global threshold |
| 14a | `step14a_pack_ppgsport.py <PPG-Sport folder>` | Optional: pack the PPG-Sport repository into one file per activity and wrist |
| 14 | `step14_ppgsport_external.py <packs or windows csv> L\|R` | MAST and the development calibrations applied directly to PPG-Sport (plan: `ANALYSIS_PLAN_PPG_SPORT.md`) |
| 15 | `step15_calibration_motion_rich.py` | Exploratory cross-sensor calibration: development and PPG-Sport persons, test on GalaxyPPG |
| 15b | `step15b_e4_control.py` | E4 control: calibration within GalaxyPPG, alone or pooled with the development persons |
| 15c | `step15c_mast_nested.py <compact>` | Nested selection of the MAST settings on a 3^4 grid |
| 16 | `step12c_crosssensor_figure.py` | Figure of the cross-sensor calibration |
| 17 | `step13_verify_numbers.py` | Recomputes the numbers quoted in the paper from `results/` and compares them |


The main steps run in a few minutes each on a laptop, except steps 1a, 3, 4a, 11 and 14, which read the raw recordings. Random draws use fixed seeds, so the results are reproduced exactly.

## Separate check

`test_causal_vs_zero_phase.py <compact>` compares the zero-phase filtering used in the paper with a causal single-pass filter (MAE and descriptors only, calibration not rerun). It is kept outside the main pipeline.

## Analysis plans

`ANALYSIS_PLAN_PPG_SPORT.md` and `ANALYSIS_PLAN_SCIENTISST_MOVE.md` were written before the corresponding data were analyzed. Each ends with a dated note on its outcome.

## Code organization

- `ppgs/data.py`: loading of PPG-DaLiA and WESAD, R-peak detection and HR per window
- `ppgs/galaxy.py`: loading of GalaxyPPG and alignment of the chest clock
- `ppgs/estimator.py`: MAST, the motion-attenuated spectral tracker (HMM tracking), and the signal descriptors
- `ppgs/conformal.py`: scale model, score functions, cross-fitting and calibrations

## License

MIT, see `LICENSE`.
