# LendingClub application scorecard model report

## Executive result

This project develops a leakage-aware application scorecard on LendingClub loans. The champion model uses monotonic WOE binning and logistic regression, with 2014–2015 as the development window, 2016 held out for model selection and 2017 reserved as an out-of-time (OOT) test. On OOT data it achieved **AUC 0.681** and **KS 0.262**. The development-to-OOT score PSI is **0.023 (stable)**.

The score is scaled to 600 points at 20:1 good-to-bad odds with PDO 50. The model base points are 461.36; a higher total score means lower predicted default risk.

## Why this is not a generic classroom scorecard

- Only application-time variables enter the model. LendingClub-assigned grade, sub-grade and interest rate are excluded as prior underwriting outputs.
- Random splitting is avoided. The 2016 validation and 2017 OOT samples reproduce the temporal drift faced by a live risk model.
- 2007–2013 legacy vintages remain in diagnostics but are excluded from fitting because fine-grained bureau-field coverage and platform policy were not comparable with later vintages.
- A validation-period intercept recalibration corrects portfolio-level PD drift while preserving rank ordering, coefficients, AUC and KS.
- 2018 observations are excluded after a vintage check reveals a sharply falling observed bad rate, consistent with incomplete outcome maturation/right censoring.
- Geography and free text are excluded because their apparent lift does not justify proxy-discrimination, instability and governance risk in a compact application scorecard.
- The report separates risk ranking from policy. Approval-rate scenarios are descriptive counterfactuals on already-granted loans, not claims about a true rejected-applicant population.

## Sample design

    | sample | start_month | end_month | observations | defaults | bad_rate |
|---|---|---|---|---|---|
| train | 2014-01-01 | 2015-12-01 | 598649 | 116966 | 19.54% |
| validation | 2016-01-01 | 2016-12-01 | 293105 | 68252 | 23.29% |
| oot | 2017-01-01 | 2017-12-01 | 169321 | 39169 | 23.13% |
| excluded_legacy_vintages | 2007-06-01 | 2013-12-01 | 227957 | 35338 | 15.50% |
| excluded_right_censored | 2018-01-01 | 2018-12-01 | 56318 | 8874 | 15.76% |

## Feature screening

Selected WOE features: **term, loan_to_income, fico_n, acc_open_past_24mths, dti, bc_open_to_buy, avg_cur_bal, loan_amnt, mths_since_recent_inq, percent_bc_gt_75, annual_inc, inq_last_6mths, num_actv_rev_tl, mort_acc, home_ownership**.

    | feature | iv |
|---|---|
| term | 0.2711 |
| loan_to_income | 0.1295 |
| fico_n | 0.1131 |
| acc_open_past_24mths | 0.0867 |
| dti | 0.0784 |
| bc_open_to_buy | 0.0577 |
| avg_cur_bal | 0.0455 |
| total_bc_limit | 0.0407 |
| loan_amnt | 0.0399 |
| mths_since_recent_inq | 0.0380 |
| percent_bc_gt_75 | 0.0334 |
| annual_inc | 0.0318 |
| inq_last_6mths | 0.0316 |
| num_actv_rev_tl | 0.0307 |
| mort_acc | 0.0297 |
| bc_util | 0.0279 |
| home_ownership | 0.0250 |
| revol_util | 0.0200 |
| purpose | 0.0160 |
| credit_history_months | 0.0134 |
| open_acc | 0.0083 |
| emp_length | 0.0070 |
| revol_bal | 0.0044 |
| pub_rec | 0.0043 |
| pub_rec_bankruptcies | 0.0036 |
| mths_since_last_major_derog | 0.0035 |
| total_bal_ex_mort | 0.0029 |
| num_accts_ever_120_pd | 0.0029 |
| delinq_2yrs | 0.0014 |
| collections_12_mths_ex_med | 0.0009 |
| num_tl_90g_dpd_24m | 0.0008 |
| pct_tl_nvr_dlq | 0.0007 |
| total_acc | 0.0003 |
| application_type | 0.0001 |

## Performance

| Sample | AUC | KS | Bad rate | Mean predicted PD | Brier score |
|---|---:|---:|---:|---:|---:|
| Development | 0.713 | 0.309 | 19.54% | 24.83% | 0.1450 |
| Validation (2016) | 0.693 | 0.275 | 23.29% | 23.29% | 0.1633 |
| OOT (2017) | 0.681 | 0.262 | 23.13% | 22.72% | 0.1655 |

## Stability monitoring

    | feature | psi | status |
|---|---|---|
| percent_bc_gt_75 | 0.1082 | watch |
| bc_open_to_buy | 0.0863 | stable |
| loan_to_income | 0.0499 | stable |
| loan_amnt | 0.0405 | stable |
| fico_n | 0.0375 | stable |
| num_actv_rev_tl | 0.0327 | stable |
| mort_acc | 0.0201 | stable |
| acc_open_past_24mths | 0.0158 | stable |
| annual_inc | 0.0119 | stable |
| home_ownership | 0.0100 | stable |
| dti | 0.0100 | stable |
| avg_cur_bal | 0.0063 | stable |
| inq_last_6mths | 0.0061 | stable |
| mths_since_recent_inq | 0.0025 | stable |
| term | 0.0012 | stable |

## Limitations and governance

The dataset contains outcomes only for loans LendingClub granted, so the model estimates default ordering inside the historical approved population. It cannot by itself learn risk for rejected applicants or prove the profitability of a new approval cutoff. Reject inference, application fraud labels, bureau freshness, income verification, LGD/EAD, operating costs and fairness testing would be required before production use. The public US P2P sample also should not be presented as a direct model for a Chinese digital bank; its value is the reproducible modelling and governance workflow.

## Reproduction

Run `python scripts/download_data.py`, then `python scripts/train_scorecard.py`. Generated artefacts include binning rules, IV decisions, coefficients, scorecard points, performance metrics, PSI tables, decile lift and approval-strategy scenarios.
