# Credit‑Card Default Prediction - Fintech 2024/25

## Dataset

[Default of credit‑card clients in Taiwan](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, Yeh & Lien 2009)
* 30,000 clients
* 25 predictor variables:

    * **Demographics**: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
    * **Credit limit**: `LIMIT_BAL`
    * **Six‑month history**:

        * Repayment status: `PAY_0`, `PAY_2` … `PAY_6`
        * Bill statements: `BILL_AMT1` … `BILL_AMT6`
        * Previous payments: `PAY_AMT1` … `PAY_AMT6`
* **Target**: `default.payment.next.month` (1 = default, 0 = non‑default)

## Project Goals

1. Train a baseline **Random‑Forest** classifier.
2. Train a **Random‑Forest preceded by L1‑based feature selection**.
3. Tune hyper‑parameters with cross‑validation (`average_precision` as the main metric) and compare the two approaches.
---

## Environment

| Package      | Version               |
| ------------ | --------------------- |
| Python       | 3.11.x                |
| scikit‑learn | ≥ 1.5                 |
| NumPy        | ≥ 1.26                |
| pandas       | ≥ 2.2                 |
| matplotlib   | ≥ 3.8 *(plots only)*  |

Create a virtual environment and install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code is fully contained in `main.py`; execute it with

```bash
python main.py
```

Everything is reproducible thanks to `random_state = 42` (set for split, CV, estimator & selector).

---

## Data preparation & preprocessing

| Step                                                                                             | Rationale                                                                                                               |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **Train/Test split** – 80 / 20, `stratify=y`                                                     | It keeps the default rate unchanged (≈ 22%) and reserves a genuine *hold-out* test.                                |
| **Monetary variables** (`BILL_AMT*`, `PAY_AMT*`) → impute 0 → `log1p(±x)` → `RobustScaler`       | Heavily skewed distributions; the log stabilises the variance, while the robust scaler reduces the impact of outliers. |
| **Payment status** (`PAY_0…PAY_6`) treated as **ordinal**                                        | Values −1…9 are delay orders; normalised with z-score, **not** one-hot to preserve order.                              |
| **Other numbers** → z‑score                                                                      | Consistency with Lasso (which requires scaled features).                                                               |
| **Categorical** (`SEX`, `EDUCATION`, `MARRIAGE`) → `OneHotEncoder(drop='first', min_frequency=2)`| Avoid extremely rare levels and collinearity.                                                                            |

All operations are wrapped in a `ColumnTransformer` **inside** every pipeline ⇒ no data‑leakage during CV.

*Note*: Scaling is technically unnecessary for trees but **required** by the Lasso selector; leaving it in both pipelines keeps them comparable.

---
## Models

### Model A - Random Forest

**Pipeline**: `preprocessor → RandomForestClassifier` -
**Hyper‑parameter search** (`RandomizedSearchCV`, 50 iterations, 5‑fold CV):

| Parameter           | Search space                      |
| ------------------- | --------------------------------- |
| `n_estimators`      | 400 … 1000 (step 100)             |
| `max_depth`         | None, 15, 20, 25, 30              |
| `max_features`      | "sqrt", "log2", 0.5, None         |
| `min_samples_split` | 2, 5, 10                          |
| `min_samples_leaf`  | 5, 10, 20                         |
| `class_weight`      | "balanced", "balanced\_subsample" |
| `max_samples`       | None, 0.8, 0.6                    |

**Scoring metric**: `average_precision` (PR‑AUC) – robust to class imbalance.

### Results

| Split     | Accuracy | Precision | Recall | F1    | ROC‑AUC | PR‑AUC | Balanced‑Acc |
| --------- | -------- | --------- | ------ | ----- | ------- | ------ | ------------ |
| **Train** | 0.831    | 0.602     | 0.699  | 0.647 | 0.901   | 0.870  | —            |
| **Test**  | 0.789    | 0.521     | 0.585  | 0.551 | 0.774   | 0.550  | 0.716        |

Confusion matrix (test):

```
TN = 3 959   FP = 714
FN = 551     TP = 776
```

---
### Model B - Lasso + Random Forest

**Pipeline**: `preprocessor → SelectFromModel(LogisticRegression L1) → RandomForestClassifier`

* **Feature selector**: Logistic Regression with `penalty = l1`, `solver = saga`, `class_weight = balanced`, `max_iter = 5000`.
  – `threshold = median` keeps roughly 50 % of non‑zero coefficients **but is also capped at `max_features = 30`**.
* **GridSearchCV** (5‑fold) jointly tunes `C` of the Lasso and RF hyper‑params (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`).

### Selected features (15 / 67)

```
BILL_AMT1, PAY_AMT1‑4, PAY_AMT6, LIMIT_BAL, AGE,
PAY_0, PAY_2‑6  (ordinal copies),
MARRIAGE_1 (married vs single/other)
```

### Results

| Split     | Accuracy | Precision | Recall | F1    | ROC‑AUC | PR‑AUC | Balanced‑Acc |
| --------- | -------- | --------- | ------ | ----- | ------- | ------ | ------------ |
| **Train** | 0.828    | 0.596     | 0.691  | 0.640 | 0.896   | 0.865  | —            |
| **Test**  | 0.788    | 0.520     | 0.580  | 0.548 | 0.776   | 0.553  | 0.714        |

Confusion matrix (test):

```
TN = 3 962   FP = 711
FN = 558     TP = 769
```

---

## Comparison

| Metric (test)         | RF full   | Lasso + RF |
| --------------------- | --------- | ---------- |
| Accuracy              | **0.789** | 0.788      |
| Precision             | **0.521** | 0.520      |
| Recall                | **0.585** | 0.580      |
| F1‑score              | **0.551** | 0.548      |
| Balanced‑Accuracy     | **0.716** | 0.714      |
| PR‑AUC                | 0.550     | **0.553**  |
| ROC‑AUC               | 0.774     | **0.776**  |
| Selected features     | 67        | **15**     |

*Statistically indistinguishable performance*: Δ(PR‑AUC) ≈ +0.003, Δ(BA) ≈ −0.002 (well within sampling error).
**Model B is therefore preferred** for its > 75 % reduction in dimensionality and easier interpretability.

For context, Preda (2021) reports a markedly lower ROC-AUC of **0.66** for a stand-alone Random Forest on the same dataset [1], highlighting how careful preprocessing and hyper-parameter tuning can lift performance by more than 10 percentage points.

---

## Next steps 
Complement the Random-Forest baselines with tree-boosting models that often excel on tabular data:

  | Model                | Validation AUC [1]    |
  | -------------------- | --------------------- |
  | `AdaBoostClassifier` | ≈ 0.65                |
  | `CatBoostClassifier` | ≈ 0.66                |
  | `XGBoost`            | **≈ 0.77**            |
  | `LightGBM`           | **≈ 0.78**            |

  > *The numbers above come from a Kaggle benchmark and serve only as guidance; re-tune on the current pipeline for a fair comparison*


---

## License

This project is released under the **MIT License**.
Dataset licensed as CC0: Public Domain.



## References

1. Preda, G. (2021). *Default of Credit Card Clients – Predictive Models*. Kaggle Notebook. (https://www.kaggle.com/code/gpreda/default-of-credit-card-clients-predictive-models)