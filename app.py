%%writefile app.py
import json, joblib, numpy as np, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Water Quality Prediction", page_icon="🌊", layout="centered")

@st.cache_resource
def load_artifacts():
    base = Path("models")
    champion = joblib.load(base / "champion.pkl")
    feature_order = joblib.load(base / "feature_order.pkl")
    label_encoder = joblib.load(base / "label_encoder.pkl")
    registry = json.loads((base / "registry.json").read_text())
    return champion, feature_order, label_encoder, registry

champion, feature_order, label_encoder, registry = load_artifacts()

st.title("🌊 Water Quality Prediction (A–E)")
st.caption("Enter parameters; missing fields are okay — the model imputes them.")

with st.expander("Model info"):
    mf1 = registry.get("metrics_macro_f1", {})
    best_f1 = max(mf1.values()) if mf1 else None
    st.write(
        f"**Champion:** {registry.get('champion','?')}"
        + (f" · **Macro‑F1:** {best_f1:.3f}" if best_f1 is not None else "")
    )
    st.write(f"**Classes:** {', '.join(registry.get('classes', []))}")

import numpy as np
with st.form("predict"):
    st.subheader("Input parameters")
    cols = st.columns(2)
    values = []
    for i, feat in enumerate(feature_order):
        with cols[i % 2]:
            txt = st.text_input(feat, value="")
            if txt.strip() == "":
                values.append(np.nan)
            else:
                try:
                    values.append(float(txt))
                except ValueError:
                    st.error(f"'{feat}' must be numeric (or leave blank).")
                    st.stop()
    go = st.form_submit_button("Predict")

if go:
    X = np.array([values])
    pred_idx = champion.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    st.success(f"**Predicted WQI Class: {pred_label}**")

    if hasattr(champion, "predict_proba"):
        proba = champion.predict_proba(X)[0]
        labels = label_encoder.inverse_transform(np.arange(len(proba)))
        st.write("Confidence by class:")
        st.bar_chart({lbl: float(p) for lbl, p in zip(labels, proba)})
    else:
        st.caption("Model does not expose probability.")
