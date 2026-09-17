# Model Card

## Intended use

This model is an educational portfolio demo for binary fraud scoring on data with the documented credit-card transaction shape (`Time`, `V1`–`V28`, `Amount`, and `Class`). It demonstrates validation, reproducible splitting, threshold selection, artifact persistence, and local inference.

## Out-of-scope uses

- Real-time production payment or other high-impact decisions.
- Claims of validity for other geographies, currencies, populations, or transaction systems.
- Per-customer fairness, risk, or explainability claims.
- Treating a score as a calibrated probability or as a substitute for investigation and policy.

## Model and training configuration

The persisted model is an sklearn `Pipeline`:

1. `StandardScaler` transforms the numeric features.
2. `LogisticRegression` uses `class_weight="balanced"`, the `liblinear` solver, and the configured iteration limit.

The default `config/config.yaml` sets `test_size: 0.2`, `validation_size: 0.2`, `split_strategy: random`, and `random_state: 42`. Training fits on train only. The threshold starts at `0.5` in configuration, then is selected on validation and persisted with the artifact.

## Evaluation protocol

The pipeline creates disjoint train, validation, and test partitions. The model is fit only on train. Candidate thresholds (including validation probabilities) are evaluated only on validation to maximize F1; ties select the highest threshold. The test partition remains untouched until final evaluation. Validation and test results are persisted separately in artifact schema `2.0`.

The metrics contract contains:

| Metric | Meaning |
| --- | --- |
| `pr_auc` | Average precision / area under the precision-recall curve. |
| `roc_auc` | Area under the ROC curve. |
| `recall` | Positive-class recall at the persisted threshold. |
| `precision` | Positive-class precision at the persisted threshold. |
| `f1` | F1 at the persisted threshold. |
| `confusion_matrix` | A 2×2 count matrix with labels `[0, 1]`. |

## Reporting policy

PR-AUC is the primary ranking metric because the positive class is rare. Report validation and test metrics separately, never as one blended result. A single public dataset cannot support production-performance claims.

## Known limitations and risks

- Severe class imbalance makes accuracy an inadequate summary and can make estimates sensitive to the split.
- PCA-transformed `V1`–`V28` features are anonymous, limiting human interpretation and practical feature review.
- The default random split can allow temporal dependence or leakage; chronological ordering and leakage controls require domain-specific decisions.
- There is no drift monitoring, calibration monitoring, alerting, retraining policy, or production incident process.
- The dataset covers two days and one historical population, so generalization is unknown.

## Monitoring and future work

A production-oriented extension would define data-quality and schema checks, monitor feature and score drift, track delayed labels and threshold performance, assess calibration, evaluate subgroup performance only with appropriate population and governance data, and establish a reviewable retraining and rollback process. None of those capabilities is claimed by this demo.
