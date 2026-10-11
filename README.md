# Router Predictive Maintenance

A machine learning project that predicts whether a network router should be **isolated** (taken out of service for a field engineer to inspect), based on its telemetry readings such as CPU, temperature, interface errors, packet drops, and alarms.

The project includes the full workflow: data cleaning, exploratory analysis, model comparison and training, threshold tuning, and an interactive Streamlit app for single-router checks and batch scoring.


---

## Table of contents

- [Problem](#problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project structure](#project-structure)
- [Setup](#setup)
- [How to reproduce the model](#how-to-reproduce-the-model)
- [Running the Streamlit app](#running-the-streamlit-app)
- [Limitations](#limitations)

---

## Problem

Routers generate many telemetry readings every hour. Engineers cannot inspect every router by hand, so the goal is to rank routers by how likely they are to need isolation, and flag the ones that need attention first.

This is a **binary classification** problem:

- **Target:** `isolate_required` (1 = isolate, 0 = healthy)
- **Class balance:** about 12.8% of routers need isolation, so the data is imbalanced. Accuracy alone is misleading, so the project focuses on precision, recall, F1, and PR-AUC.

## Dataset

The raw telemetry file had 3,090 rows and 19 columns. Cleaning produced 2,995 rows and 17 columns (16 features plus the target).

| Feature type | Columns |
|---|---|
| Numeric (11) | `cpu_percent`, `memory_percent`, `heap_percent`, `temperature_c`, `fan_speed_rpm`, `power_draw_w`, `interface_error_count`, `packet_drop_rate`, `latency_ms`, `uptime_hours`, `alarm_count` |
| Categorical (5) | `vendor`, `router_model`, `slot_id`, `card_status`, `link_status` |
| Target | `isolate_required` |

`router_id` and `ip_address` were dropped because they identify routers and do not describe their health.

The dataset itself is **not included in this repository** (see [Limitations](#limitations)). The app and training scripts expect it at `data/cleaned_router_telemetry_dataset.csv`.

## Approach

### 1. Data cleaning (`notebooks/data_preprocessing.ipynb`)

- Removed 90 exact duplicate rows (3,090 to 3,000).
- Detected outliers in each numeric column using the IQR rule (values beyond 1.5 × IQR from the quartiles).
- Removed rows that were outliers in at least 6 of the 11 numeric features (5 rows removed).
- Converted text columns to a consistent type.

### 2. Exploratory data analysis (`notebooks/eda.ipynb`)

- Class distribution and feature distributions.
- Correlation heatmap, checking for highly correlated feature pairs.
- Correlation of each feature with the target. The strongest relationships were with `packet_drop_rate`, `interface_error_count`, `latency_ms`, and `alarm_count`, along with `card_status` and `link_status`.

### 3. Splitting the data (`src/train.py`)

The cleaned data is split into three parts, stratified so each part keeps the same share of isolated routers:

| Split | Rows | Purpose |
|---|---|---|
| Train (80%) | 2,396 | Model fitting and cross-validation |
| Validation (10%) | 299 | Choosing the decision threshold |
| Holdout (10%) | 300 | Final, untouched performance check |

The split uses `random_state = 42`, so it is reproducible.

### 4. Model training (`src/train.py`)

- **Preprocessing (inside the pipeline):** numeric features get median imputation and standard scaling. Categorical features get most-frequent imputation and one-hot encoding.
- **Class imbalance:** SMOTE oversampling, applied only to training folds during cross-validation, so no synthetic rows leak into validation.
- **Models compared:** Logistic Regression, Random Forest, SVC, XGBoost, and K-Nearest Neighbours.
- **Search:** grid search over each model's settings, with 5-fold stratified cross-validation, optimizing F1.
- **Threshold tuning:** the model outputs a score from 0 to 1. The decision threshold is chosen on the validation set to maximize F1. Routers scoring at or above the threshold are flagged.
- **Final check:** the holdout set is used once, to report performance.

All candidate models and their scores are saved to `models/model_comparison_cv.csv`.

## Results

Numbers from the training run saved in `models/training_metadata.json`.

**Selected model:** XGBoost (`learning_rate = 0.01`, `max_depth = 5`, `n_estimators = 200`, `subsample = 1.0`, `colsample_bytree = 1.0`)

**Selected threshold:** 0.65

### Holdout set (final check)

| Threshold | Precision | Recall | F1 | Accuracy | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Default (0.50) | 0.38 | 0.72 | 0.50 | 0.81 | 0.87 | 0.62 |
| Tuned (0.65) | 0.64 | 0.41 | 0.50 | 0.89 | 0.87 | 0.62 |

**How to read this:** at the default threshold, the model catches about 72% of routers that need isolation, but about 6 in 10 of its flags are false alarms. At the tuned threshold, most flags are correct (precision 0.64), but it catches only about 41% of the routers that need isolation. The threshold is a trade-off between missed failures and false alarms. The tuned threshold was chosen because false alarms cost field engineer time.

### Validation set

| Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Default (0.50) | 0.39 | 0.76 | 0.51 | 0.88 | 0.60 |
| Tuned (0.65) | 0.72 | 0.47 | 0.57 | 0.88 | 0.60 |

## Project structure

```
router_predictive_maintenance/
├── data/                              # Telemetry data (not included in this repo)
│   ├── router_telemetry_dataset.csv           # Raw data
│   ├── cleaned_router_telemetry_dataset.csv   # Output of data cleaning
│   ├── train_80.csv                           # Created by train.py
│   └── validation_10.csv                      # Created by train.py
├── models/
│   ├── best_model.joblib              # Trained model pipeline
│   ├── training_metadata.json         # Metrics, settings, and thresholds from the last run
│   └── model_comparison_cv.csv        # Cross-validation scores for every model tried
├── notebooks/
│   ├── data_preprocessing.ipynb       # Cleaning steps
│   └── eda.ipynb                      # Exploratory analysis
├── src/
│   ├── train.py                       # Trains, tunes, and saves the model
│   └── app.py                         # Streamlit app
├── requirements.txt
└── README.md
```

## Setup

**Requirements:** Python 3.10 or newer.

1. Clone the repository and open the project folder.
2. Install the packages:

   ```
   python -m pip install -r requirements.txt
   ```

3. Place the cleaned dataset at `data/cleaned_router_telemetry_dataset.csv`. It is not included in this repository.

## How to reproduce the model

Run the steps in this order:

1. **Clean the data** (only needed if you start from the raw file): open `notebooks/data_preprocessing.ipynb` and run all cells. This writes the cleaned dataset to `data/`.
2. **Explore (optional):** open `notebooks/eda.ipynb` and run all cells.
3. **Train:** from the project root, run:

   ```
   python src/train.py
   ```

   This takes a few minutes. It saves the model to `models/`, the metrics to `models/training_metadata.json`, and the model comparison to `models/model_comparison_cv.csv`.

Results can differ slightly between library versions. The metadata file records what your run produced.

## Running the Streamlit app

From the project root, run:

```
streamlit run src/app.py
```

The app opens in your browser at `http://localhost:8501`. It has three tabs:

- **Single router:** load a healthy router, a failing router, or a random router from the data. Adjust the sliders and dropdowns to see the score and the HEALTHY or ISOLATION REQUIRED result change.
- **Batch CSV:** download a sample file, or upload your own file with one row per router. The app returns a ranked table and a list of routers to isolate, which you can download as CSV. Your file needs the 16 feature columns listed above. Extra columns, such as `router_id`, are kept in the output.
- **Model performance:** shows the metrics from the last training run, comparing the default and tuned thresholds.

The app loads the saved model once at startup, so it does not retrain the model.

## Limitations

- **Label definition.** In this data, failed cards are always flagged and down links are flagged 90% of the time. Much of the signal comes from simple status rules, so the model is partly learning those rules. The way `isolate_required` was labelled is not documented in this repository.
- **Score is not a probability.** The score ranks routers by how strongly the model flags them. It is not a calibrated chance that a router will fail.
- **Single dataset.** The model was trained and tested on one dataset from one network. Performance on other networks, vendors, or time periods is unknown.
- **Demo samples come from training data.** The random routers and sample batch in the app come from the same data the model was trained on, so their scores look better than they would on new routers.
- **No live data feed.** The app works on files you upload. It does not connect to live network equipment.
- **Some sensor readings look suspicious.** A few readings are exactly 0 (for example `cpu_percent` and `heap_percent`). They may be sensor dropouts and were kept in the data.
