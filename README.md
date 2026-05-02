# SAINT-IDS: Experimental Data and Evaluation Results

[![DOI](https://zenodo.org/badge/DOI/XXXX.svg)](https://doi.org/XXXX)

## Overview

This repository contains the experimental data and evaluation results supporting the paper:

> **SAINT-IDS: A Privacy-Preserving Real-Time Intrusion Detection System for Healthcare IoT Networks**
>
> Djamel Eddine Benzellat and Asma Amraoui
>
> *Concurrency and Computation: Practice and Experience* (CCPE), 2026.

SAINT-IDS is a real-time, edge-deployed intrusion detection system that achieves metadata-only privacy preservation for healthcare IoT networks through a dual-model ensemble (Isolation Forest + LightGBM), CKKS homomorphic encryption for cloud aggregation, and healthcare protocol-aware detection.

## Repository Structure

```
SAINT-IDS-Data/
├── README.md                              ← This file
├── training/
│   └── training_report.json               ← Model training pipeline output
├── evaluation/
│   ├── ablation_study.json                ← Ensemble weight ablation study
│   ├── healthcare_evasion_test.json       ← Healthcare protocol detection tests
│   └── ckks_benchmarks.json               ← CKKS encryption latency benchmarks
└── figures/
    ├── confusion_matrix.pdf               ← LightGBM confusion matrix (Figure 3)
    ├── per_class_metrics.pdf              ← Per-class precision/recall/F1 (Figure 4)
    ├── training_curves.pdf                ← Training convergence curves (Figure 5)
    └── training_curves_data.json          ← Raw data behind Figure 5
```

## File Descriptions

### `training/training_report.json`

Complete output of the ML training pipeline. Contains:

- **Dataset information**: CICIDS-2017 files used, class distribution after balancing (2,830,743 samples)
- **Optuna hyperparameters**: Best configuration from 50-trial Bayesian search (max\_depth=10, learning\_rate=0.039, n\_estimators=479, num\_leaves=130)
- **LightGBM metrics**: Per-class precision, recall, F1-score; weighted F1=0.9986; FPR=0.16%
- **Confusion matrix**: 7×7 matrix for all attack classes
- **Feature importance**: Top-10 features by split importance
- **Isolation Forest metrics**: Anomaly precision, recall, F1 (standalone)
- **Validation gates**: Pass/fail status for all 5 deployment gates
- **ONNX export benchmarks**: FP32 and INT8 inference latency, throughput (flows/sec)

**Referenced in**: Tables 3, 4, 5; Figures 3, 4, 5 of the paper.

### `evaluation/ablation_study.json`

Ablation study comparing four ensemble fusion weight configurations (A–D) on the held-out CICIDS-2017 test set (424,612 samples). Contains:

- **Configurations**: LightGBM-only (1.0/0.0), 90/10, 70/30, 50/50
- **Per-config metrics**: Classification F1, FPR, alert-level TPR/FPR, per-class F1 and recall
- **Alert thresholds**: Critical=0.85, Warning=0.50

**Referenced in**: Table 8 and Section VI-F of the paper.

### `evaluation/healthcare_evasion_test.json`

Results from 16 synthetic healthcare protocol test scenarios targeting DICOM (ports 104, 11112), HL7 (port 2575), FHIR (port 8443), and CoAP (port 5683). Contains:

- **Per-scenario results**: Name, category, port, expected vs. actual class, severity, score, detection status
- **Summary statistics**: 14/14 attacks detected, 2/2 baselines clean, 100% detection rate, 85.7% classification accuracy
- **Categories tested**: FastFilter (4), RuleOverlay (3), Evasion (5), ML Classification (2), Baselines (2)

**Referenced in**: Tables 6, 7 and Section VI-E of the paper.

### `evaluation/ckks_benchmarks.json`

CKKS homomorphic encryption pipeline latency benchmarks measured over 100 iterations on the 28-field sanitized feature vector. Contains:

- **Per-stage latency**: PHI Sanitizer, CKKS Encrypt+HMAC, CKKS Decrypt (p50, p95, p99, mean in ms)
- **Accuracy loss**: Mean absolute error = 1.95 × 10⁻⁷, classification agreement = 100%
- **Full pipeline**: Median = 0.084 ms, p99 = 0.437 ms

**Referenced in**: Table 4 and Section VI-C of the paper.

### `figures/`

Publication-quality figures in PDF format, plus the raw JSON data used to generate the training convergence curves.

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `confusion_matrix.pdf` | LightGBM classifier confusion matrix on held-out test set (424,612 samples) | Figure 3 |
| `per_class_metrics.pdf` | Per-class precision, recall, and F1-score bar chart | Figure 4 |
| `training_curves.pdf` | LightGBM training and validation loss/F1 convergence curves | Figure 5 |
| `training_curves_data.json` | Raw per-iteration loss and F1 values used to generate Figure 5 | Figure 5 (source data) |

## Dataset

The primary training dataset is **CICIDS-2017**, publicly available from the Canadian Institute for Cybersecurity:

> Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*, 108–116. DOI: [10.5220/0006639801080116](https://doi.org/10.5220/0006639801080116)
>
> Download: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

All eight capture files (Monday–Friday) were used for training. The seven-class attack taxonomy mapping is described in Section V-A of the paper.

## Experimental Environment

- **Hardware**: Intel Core processor, 16 GB RAM
- **Software**: Python 3.12.2, scikit-learn 1.4, LightGBM 4.3, Optuna 3.5, ONNX Runtime 1.17
- **Deployment**: Docker Engine 25.0, containerized edge stack

## License

This data is provided under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Citation

If you use this data, please cite:

```bibtex
@article{benzellat2026saintids,
  author  = {Benzellat, Djamel Eddine and Amraoui, Asma},
  title   = {{SAINT-IDS}: A Privacy-Preserving Real-Time Intrusion Detection System for Healthcare {IoT} Networks},
  journal = {Concurrency and Computation: Practice and Experience},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

## Contact

- **Djamel Eddine Benzellat** — [djameleddine.benzellat@univ-tlemcen.dz](mailto:djameleddine.benzellat@univ-tlemcen.dz) — ORCID: [0009-0007-3388-7938](https://orcid.org/0009-0007-3388-7938)
- **Asma Amraoui** — ORCID: [0000-0001-7020-6846](https://orcid.org/0000-0001-7020-6846)

Telecommunications Laboratory, University of Tlemcen, Algeria
