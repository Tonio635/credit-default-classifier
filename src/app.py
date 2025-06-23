import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
import matplotlib.pyplot as plt
import math

def signed_log_transform(X):
    """
    Apply signed log transformation to the input data.
    """
    return np.sign(X) * np.log1p(np.abs(X))

# ------------------------------------------------------------
# Credit‑Default Predictor – Streamlit app
# ------------------------------------------------------------
MODEL_DIR = Path("./assets/models")
MODEL_FULL_PATH = MODEL_DIR / "rf_full.joblib"
MODEL_LASSO_PATH = MODEL_DIR / "rf_lasso.joblib"
TEST_SPLIT_PATH = MODEL_DIR / "test_split.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "rf_lasso_features.pkl"

# ------------------------------------------------------------
# Cached loaders (executed once per session)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_resources():
    """Load models + hold-out split from disk (cached)."""
    model_full = joblib.load(MODEL_FULL_PATH)
    model_lasso = joblib.load(MODEL_LASSO_PATH)
    X_test, y_test = pickle.load(open(TEST_SPLIT_PATH, "rb"))
    try:
        selected_feats = pickle.load(open(FEATURE_LIST_PATH, "rb"))
    except FileNotFoundError:
        selected_feats = None
    return model_full, model_lasso, X_test, y_test, selected_feats

model_full, model_lasso, X_test, y_test, selected_feats = load_resources()

# ------------------------------------------------------------
# 1. Probabilities computation (cached per session)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_probs():
    """Returns a dict {model_name: y_prob_array}."""
    return {
        name: mdl.predict_proba(X_test)[:, 1]
        for name, mdl in [("RF full", model_full), ("RF Lasso", model_lasso)]
    }

probs_dict = get_probs()

# ------------------------------------------------------------
# 2. Metrics computation (cached per session)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_metrics(threshold: float) -> pd.DataFrame:
    rows = []
    for name, y_prob in probs_dict.items():
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
            "Balanced-Acc": balanced_accuracy_score(y_test, y_pred),
        })
    return pd.DataFrame(rows).set_index("Model").round(3)

# ------------------------------------------------------------
# Streamlit UI configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Credit-Default Predictor", layout="wide")
st.title("Credit-Default Predictor")

# ------------------------------------------------------------------
# Sidebar – global controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Global settings")
    model_choice = st.radio(
        "Model:",
        (
            "Random Forest - All features",
            "Random Forest - Selected features (Lasso)",
        ),
    )
    model = model_full if model_choice.startswith("Random Forest - All") else model_lasso
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)

metrics_df = compute_metrics(threshold)

# ------------------------------------------------------------------
# Helper – column sets & widgets
# ------------------------------------------------------------------
orig_cols = model.named_steps["prep"].feature_names_in_

if selected_feats is not None:
    sel_base_cols = set()
    for feat in selected_feats:
        part = feat.split("__", 1)[1] if "__" in feat else feat
        base = part.split("_")[0] if feat.startswith("cat__") else part
        sel_base_cols.add(base)
else:
    sel_base_cols = set(orig_cols)

visible_cols = list(orig_cols) if model is model_full else [c for c in orig_cols if c in sel_base_cols]

df_ref = X_test  # for widget ranges
cat_small = {"SEX", "EDUCATION", "MARRIAGE"}


def numeric_step(series: pd.Series) -> float:
    """Compute a float step size compatible with Streamlit number_input."""
    rng = series.max() - series.min()
    if pd.isna(rng) or rng == 0:
        return 1.0
    exp = max(int(math.log10(rng)) - 2, 0)
    return float(10 ** exp)


def widget_for(col: str):
    """Create a widget for *col* and return user input."""
    if col in cat_small:
        opts = sorted(df_ref[col].dropna().unique())
        return st.selectbox(col, opts, key=f"w_{col}")
    if col.startswith("PAY_") and col[4:].isdigit():
        return st.slider(col, -2, 8, 0, 1, key=f"w_{col}")
    series = df_ref[col].dropna()
    if series.empty:
        return st.number_input(col, value=0.0, step=1.0, key=f"w_{col}")
    q05, q95 = series.quantile([0.05, 0.95]).astype(float)
    if q05 == q95:
        q05 -= 1.0; q95 += 1.0
    default = float(series.median())
    step_val = numeric_step(series)
    return st.number_input(col, float(q05), float(q95), default, step_val, key=f"w_{col}")

# defaults: let pipeline imputer handle NaN
nan_default = {col: np.nan for col in orig_cols}

# ------------------------------------------------------------------
# Main tabs
# ------------------------------------------------------------------
TAB_SINGLE, TAB_BATCH, TAB_INSIGHTS = st.tabs(["🔮 Single", "📂 Batch", "📊 Insights"])

# ------------------------------------------------------------
# Single prediction
# ------------------------------------------------------------
with TAB_SINGLE:
    st.subheader("Prediction for a single user")
    with st.form("user_form"):
        user_vals = {col: widget_for(col) for col in visible_cols}
        submitted = st.form_submit_button("Predict")
    if submitted:
        full_input = nan_default.copy()
        full_input.update(user_vals)
        df_single = pd.DataFrame([full_input])[orig_cols]
        prob = model.predict_proba(df_single)[0, 1]
        pred = int(prob >= threshold)
        st.metric("Probability of default", f"{prob:.3f}")
        st.write("Prediction:", "🟥 Default" if pred else "🟩 No default")
        st.caption(f"Visible features: {len(visible_cols)} / {len(orig_cols)}")

# ------------------------------------------------------------
# Batch scoring
# ------------------------------------------------------------
with TAB_BATCH:
    st.subheader("Batch scoring da CSV")
    uploaded = st.file_uploader("Carica un CSV con le colonne del training", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded, sep=";")
        for col in orig_cols:
            if col not in df_batch.columns:
                df_batch[col] = np.nan
        df_batch = df_batch[orig_cols]
        probs = model.predict_proba(df_batch)[:, 1]
        df_batch["prob_default"] = np.round(probs, 3)
        df_batch["prediction"] = (probs >= threshold).astype(int)
        st.success("✓ Predizioni calcolate")
        st.dataframe(df_batch.head(50))
        st.download_button("📥 Scarica tutte le predizioni", df_batch.to_csv(index=False).encode(), "predizioni.csv", mime="text/csv")
    else:
        st.info("Upload a CSV file to start batch scoring.")

# ------------------------------------------------------------
# Model insights
# ------------------------------------------------------------
with TAB_INSIGHTS:
    st.subheader("Summary metrics")

    sel_name = "RF full" if model is model_full else "RF Lasso"
    st.dataframe(
        metrics_df.loc[[sel_name]],
        use_container_width=True,
    )
    st.caption(
        f"Selected model: **{sel_name}** &nbsp;|&nbsp; "
        f"Decisional threshold: **{threshold:.2f}**"
    )

    st.subheader("Curves and matrices - Selected model")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("ROC curve")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2)
        ax2.set_title("PR curve")
        st.pyplot(fig2)

    st.subheader("Confusion matrix")
    fig3, ax3 = plt.subplots()
    y_prob_cm = probs_dict[sel_name]
    y_pred_cm = (y_prob_cm >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cm, ax=ax3)
    st.pyplot(fig3)

    if hasattr(model.named_steps["rf"], "feature_importances_"):
        st.subheader("Top-20 feature importances")
        imps = model.named_steps["rf"].feature_importances_
        fn = model.named_steps["prep"].get_feature_names_out()
        idx_top = np.argsort(imps)[::-1][:20]
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.barh(np.array(fn)[idx_top][::-1], imps[idx_top][::-1])
        ax4.set_xlabel("Importance")
        st.pyplot(fig4)

    if selected_feats is not None and model_choice.endswith("Lasso)"):
        st.subheader("Lasso selected features (post-encoding)")
        st.code("\n".join(selected_feats))

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Streamlit App")
