import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st


# ---------- Paths (relative to this file, so the app runs from any folder) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_router_telemetry_dataset.csv"
TARGET = "isolate_required"

NUMERIC_COLS = [
    "cpu_percent", "memory_percent", "heap_percent", "temperature_c",
    "fan_speed_rpm", "power_draw_w", "interface_error_count",
    "packet_drop_rate", "latency_ms", "uptime_hours", "alarm_count",
]
CATEGORICAL_COLS = ["vendor", "router_model", "slot_id", "card_status", "link_status"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


# ---------- Loading (cached so the app stays fast) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def score(model, X):
    """Returns the model's isolation score (0 to 1) for each row."""
    return model.predict_proba(X[FEATURE_COLS])[:, 1]


def score_batch(model, raw, threshold):
    """Scores an uploaded table. Raises ValueError if required columns are missing."""
    missing = [c for c in FEATURE_COLS if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    result = raw.copy()
    result["isolation_score"] = score(model, raw).round(4)
    result["action"] = np.where(result["isolation_score"] >= threshold, "ISOLATE", "OK")
    return result.sort_values("isolation_score", ascending=False)


def profile_from(group):
    """Builds a typical telemetry profile from a group of rows (median for numbers, most common for text)."""
    values = {c: float(group[c].median()) for c in NUMERIC_COLS}
    values.update({c: group[c].mode().iloc[0] for c in CATEGORICAL_COLS})
    return values


# ---------- Page ----------
st.set_page_config(page_title="Router Isolation Predictor", layout="wide")
st.title("Router Isolation Predictor")
st.caption(
    "Estimates whether a router should be isolated, based on its telemetry. "
    "Built with XGBoost on router telemetry data."
)

model = load_model()
data = load_data()
meta = load_metadata()
THRESHOLD = float(meta["selected_threshold_for_inference"])

healthy = profile_from(data[data[TARGET] == 0])
failing = profile_from(data[data[TARGET] == 1])


def set_values(values):
    for key, value in values.items():
        st.session_state[key] = value


def load_random_router():
    row = data.sample(1).iloc[0]
    set_values({c: row[c].item() if hasattr(row[c], "item") else row[c] for c in FEATURE_COLS})


# Start every input at the "healthy" profile the first time the page loads.
for col in FEATURE_COLS:
    if col not in st.session_state:
        st.session_state[col] = healthy[col]


tab_single, tab_batch, tab_model = st.tabs(["Single router", "Batch CSV", "Model performance"])


# ---------- Tab 1: single router ----------
with tab_single:
    st.subheader("Try a router")
    b1, b2, b3 = st.columns(3)
    b1.button("Load healthy router", on_click=set_values, args=(healthy,), width="stretch")
    b2.button("Load failing router", on_click=set_values, args=(failing,), width="stretch")
    b3.button("Load random router from data", on_click=load_random_router, width="stretch")

    left, right = st.columns([3, 2])

    with left:
        num_cols = st.columns(2)
        for i, col in enumerate(NUMERIC_COLS):
            lo, hi = float(data[col].min()), float(data[col].max())
            is_int = pd.api.types.is_integer_dtype(data[col])
            with num_cols[i % 2]:
                if is_int:
                    st.slider(col, int(lo), int(hi), key=col, step=1)
                else:
                    st.slider(col, lo, hi, key=col, step=0.1)

        cat_cols = st.columns(len(CATEGORICAL_COLS))
        for i, col in enumerate(CATEGORICAL_COLS):
            options = sorted(data[col].astype(str).unique().tolist())
            with cat_cols[i]:
                st.selectbox(col, options, key=col)

    with right:
        X_one = pd.DataFrame([{c: st.session_state[c] for c in FEATURE_COLS}])
        one_score = float(score(model, X_one)[0])

        st.metric("Estimated isolation score", f"{one_score:.1%}")
        if one_score >= THRESHOLD:
            st.error("ISOLATION REQUIRED")
        else:
            st.success("HEALTHY")
        st.caption(
            f"Routers scoring at or above {THRESHOLD:.2f} are flagged. "
            "The score is the model's estimate, not a calibrated probability."
        )


# ---------- Tab 2: batch ----------
with tab_batch:
    st.subheader("Score many routers at once")
    st.write(
        "Upload a CSV with one row per router. It needs these columns: "
        + ", ".join(FEATURE_COLS) + ". Extra columns (like router_id) are kept in the output."
    )

    sample = data[FEATURE_COLS].sample(50, random_state=1).to_csv(index=False)
    st.download_button(
        "Download sample batch (50 routers)",
        data=sample,
        file_name="sample_telemetry_batch.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload telemetry CSV", type="csv")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            result = score_batch(model, raw, THRESHOLD)
        except ValueError as e:
            st.error(str(e))
        except Exception:
            st.error("Could not score this file. Check that the numeric columns contain only numbers.")
        else:
            flagged = result[result["action"] == "ISOLATE"]
            c1, c2 = st.columns(2)
            c1.metric("Routers scored", len(result))
            c2.metric("Flagged for isolation", len(flagged))

            st.dataframe(result, width="stretch", hide_index=True)

            d1, d2 = st.columns(2)
            d1.download_button(
                "Download all results (CSV)",
                data=result.to_csv(index=False),
                file_name="scored_routers.csv",
                mime="text/csv",
            )
            d2.download_button(
                "Download isolation queue (CSV)",
                data=flagged.to_csv(index=False),
                file_name="isolation_queue.csv",
                mime="text/csv",
            )


# ---------- Tab 3: model performance ----------
with tab_model:
    st.subheader("How well the model performs")
    st.write(
        "Numbers come from the training run (saved in training_metadata.json). "
        "The decision threshold controls the trade-off: a higher threshold means fewer false alarms "
        "but more missed failing routers."
    )

    rows = []
    for split in ["validation", "holdout"]:
        default_m = meta[f"{split}_metrics_default_threshold"]
        tuned_m = meta[f"{split}_metrics_tuned_threshold"]
        for label, m, t in [("Default (0.50)", default_m, 0.5), (f"Tuned ({THRESHOLD:.2f})", tuned_m, THRESHOLD)]:
            rows.append({
                "Split": split.capitalize(),
                "Threshold": label,
                "Precision": round(m["precision"], 3),
                "Recall": round(m["recall"], 3),
                "F1": round(m["f1_score"], 3),
                "Accuracy": round(m["accuracy"], 3),
                "ROC-AUC": round(m["roc_auc"], 3),
                "PR-AUC": round(m["pr_auc"], 3),
            })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("**Confusion matrix at the tuned threshold (holdout set)**")
    cm = meta["holdout_metrics_tuned_threshold"]["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actually healthy", "Actually isolated"],
        columns=["Predicted healthy", "Predicted isolate"],
    )
    st.dataframe(cm_df, width="stretch")

    class_rate = meta["class_balance"]["holdout_positive_rate"]
    st.caption(
        f"Only about {class_rate:.0%} of routers need isolation, so accuracy alone is misleading. "
        "PR-AUC and recall matter more here."
    )