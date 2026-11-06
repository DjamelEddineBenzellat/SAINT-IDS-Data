# Revision 1 (R1) evidence

This folder adds the evidence for the revised manuscript (CCPE, revision 1). The data and figure files from the
original submission are kept unchanged for traceability; the ones the revision withdraws are listed below.

## What is new

| File | Where it is used in the revised paper |
|---|---|
| `evaluation_r1/classification_v2_training_report.json` | Section 5.1, Table 5, Figures 3-5 |
| `evaluation_r1/threshold_test_eval_tau_0.25.json` | Section 6.3 (alert threshold on the test split) |
| `evaluation_r1/threshold_tune_meta.json` | Table 4 (cost ratio and threshold, tune split) |
| `evaluation_r1/ckks_benchmarks_tenseal.json` | Table 7 (real CKKS cost) |
| `evaluation_r1/latency_stages_run1.json` | Table 8 (per-stage latency, deployed code) |
| `evaluation_r1/latency_stages_run2.json` | Table 8 |
| `evaluation_r1/latency_stages_run3.json` | Table 8 |
| `evaluation_r1/latency_stages_default_onnx_run1.json` | Section 5.2 (default ONNX Runtime session options) |
| `evaluation_r1/latency_stages_default_onnx_run2.json` | Section 5.2 |
| `evaluation_r1/latency_stages_default_onnx_run3.json` | Section 5.2 |
| `evaluation_r1/latency_stages_summary.json` | Table 8, Section 5.2 (medians of 3 runs) |
| `evaluation_r1/load_test_http.json` | Table 9 (full system over HTTP) |
| `evaluation_r1/load_test_components_first_series.json` | Table 14 (component cost), first load series |
| `evaluation_r1/components_in_process.json` | Table 14 footnote (in-process cost) |
| `evaluation_r1/scaling_diagnostic.json` | Section 6.5 (scaling) |
| `evaluation_r1/isolation_forest_known_attacks.json` | Table 15 (Isolation Forest, known attacks) |
| `evaluation_r1/isolation_forest_unseen_classes.json` | Table 16 (unseen attack classes) |
| `evaluation_r1/shap_method_comparison.json` | Section 4.4 |
| `evaluation_r1/privacy_residual_risk.json` | Table 10 (residual re-identification risk) |
| `evaluation_r1/content_attack_recall.json` | Table 17 (content-type attacks) |
| `evaluation_r1/ciciomt2024_extractor_validation.json` | Section 6.7 (flow extractor validation) |
| `evaluation_r1/ciciomt2024_extractor_config.json` | Section 6.7 |
| `evaluation_r1/ciciomt2024_deployed_detector.json` | Tables 11-12 (false alarms, detection and naming) |
| `evaluation_r1/ciciomt2024_retrained_models.json` | Table 13 (retrained research models) |
| `figures_r1/table4_feature_importance_v2.json` | Table 6 |
| `figures_r1/training_curves_v2.json` | Figure 5 |
| `figures_r1/figure1_r1.pdf` | Figure 1 |
| `figures_r1/figure2_r1.pdf` | Figure 2 |
| `figures_r1/confusion_matrix_v2.pdf` | Figure 3 |
| `figures_r1/per_class_metrics_v2.pdf` | Figure 4 |
| `figures_r1/training_curves_v2.pdf` | Figure 5 |

`MANIFEST_R1.json` gives the SHA-256 of every file.

## Analysis scripts (`scripts_r1/`)

These are the scripts that produced the R1 results, as they were run.

**Folders:**
- `analysis/`: one script per experiment;
- `run/`: the shell drivers that ran them, one folder per experiment batch;
- `training/`: the deduplicated training protocol and its configuration;
- `figures/`: the figure generators.

**Before you run them:**
- Local directories are written as `/path/to/SAINT` (the checkout) and `/path/to/workdir` (the working directory of the long runs); replace them with your own.
- The scripts import the SAINT-IDS edge code (`edge.*`, `contracts.*`), which is not part of this repository.
- They need the two datasets above.
- Keys and HMAC secrets in the scripts are test values for load tests and parity runs, not deployment keys. IP addresses in the scripts are synthetic test addresses.

| Script | Result files |
|---|---|
| `training/train_pipeline_v2.py`, `training/dedup.py`, `training/configs/v2_dedup.json` | `classification_v2_training_report.json`, `figures_r1/training_curves_v2.json` |
| `analysis/c5_threshold_sweep_v2.py` | `threshold_tune_meta.json`, `threshold_test_eval_tau_0.25.json` |
| `analysis/c10_ckks_bench.py` | `ckks_benchmarks_tenseal.json` |
| `analysis/stage_latency_r1.py`, `analysis/stage_r1_analysis.py` | `latency_stages_run{1,2,3}.json`, `latency_stages_default_onnx_run{1,2,3}.json`, `latency_stages_summary.json` |
| `analysis/c9_latency_e4.py` | an earlier stage-latency run with default ONNX Runtime options, replaced by `stage_latency_r1.py`; its results are not published |
| `analysis/x2_run.py`, `x2_app.py`, `x2_load.py`, `x2_receiver.py` | `load_test_http.json`, `load_test_components_first_series.json` |
| `analysis/x3_offline.py` | `components_in_process.json` |
| `analysis/diag_b.py`, `diag_app.py` | `scaling_diagnostic.json` |
| `analysis/c8_step1_evidence.py` | `isolation_forest_known_attacks.json`, `isolation_forest_unseen_classes.json`, `shap_method_comparison.json` |
| `analysis/c11_linkage.py` | `privacy_residual_risk.json` |
| `analysis/c3_rawlabel_recall.py` | `content_attack_recall.json` |
| `analysis/c12_flowmeter.py`, `c12_common.py`, `c12_e0.py` (`c12_e0b.py`, `c12_selftest.py`: diagnostics and self-test) | `ciciomt2024_extractor_validation.json`, `ciciomt2024_extractor_config.json` |
| `analysis/c12_inventory.py`, `c12_e1.py` (`c12_e1_diag.py`: diagnostic) | `ciciomt2024_deployed_detector.json` |
| `analysis/c12_e1b.py` | `ciciomt2024_retrained_models.json` |
| `figures/generate_figures_r1.py`, `figures/generate_diagrams_r1.py` | `figures_r1/*.pdf`, `figures_r1/table4_feature_importance_v2.json` |
| `analysis/b1_comment8_audit.py`, `b2_comment6_audit.py`, `c1_policy_harness.py`, `c2_threshold_sweep.py`, `c3_healthcare_replay_cprime.py`, `c4_comment8_v1_vs_v2.py`, `c6_healthcare_replay_v2.py`, `c7_latency_v2.py` | audits of the original submission and of the alert policy (data integrity, alert rate, threshold, healthcare scenarios, latency) |
| `analysis/hw_monitor.sh`, `thermal_guard.sh` | hardware logging during the long runs |

**Later versions of two scripts:** two of the published scripts are later versions than the ones that produced some results. Neither change affects those results.
- **`x2_run.py`:** the first load series (`load_test_components_first_series.json`) ran an earlier version of this script. The published version adds a `--no-x3` option and sets a test IP-pseudonym key.
- **`c12_e1.py`:** the published version adds an optional argument that labels only selected flow tables.

**Per-source limit:** when these results were produced, the alert router still counted scored flows against a limit of 100 per minute per source address. It was removed afterwards, so the paper describes a router without it. No published result depends on it: `x2_load.py` spreads requests over 4,096 synthetic source addresses so that the limit never triggered, and the other scripts raise it with `max_alerts_per_minute`. To run those scripts on the current code, drop that argument.

## Files from the original submission that the revision supersedes

| Original file | Status in R1 | Reason |
|---|---|---|
| `evaluation/ckks_benchmarks.json` | superseded by `evaluation_r1/ckks_benchmarks_tenseal.json` | produced by a TenSEAL-compatible simulator that performs no CKKS arithmetic |
| `evaluation/ablation_study.json` | superseded by `evaluation_r1/isolation_forest_*.json` and `load_test_components_first_series.json` | the ensemble score it used alerted on almost every benign flow |
| `evaluation/healthcare_evasion_test.json` | superseded by `evaluation_r1/ciciomt2024_*.json` | 16 synthetic scenarios; their detections came from rules, not the model |
| `training/training_report.json` | superseded by `evaluation_r1/classification_v2_training_report.json` | the original split had duplicate flows across training and test; R1 deduplicates before splitting |
| `figures/*.pdf`, `figures/training_curves_data.json` | superseded by `figures_r1/` | regenerated from the deduplicated model; the original Figure 5 came from a separate run |

## Not included

- **The datasets themselves:**
  - CICIDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html
  - CICIoMT2024: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html (registration required)
- **Model binaries and keys.**
- **IP addresses:** all were removed from the result files, including the CICIoMT2024 testbed hosts.
