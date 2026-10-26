"""
app.py - Streamlit front end for the General Disease Prediction System.
Run with:  streamlit run app.py
"""
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import config
from config import DISEASES, MODELS_DIR

st.set_page_config(page_title="General Disease Prediction System", page_icon="🩺",
                   layout="wide")

DISCLAIMER = ("⚠️ **Educational prototype - NOT a medical diagnostic tool.** Predictions come "
              "from small public datasets and must never be used for real medical decisions. "
              "Always consult a qualified doctor.")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

st.markdown("""
<style>
.hero {padding: 2rem 2rem; border-radius: 14px; color: white;
       background: linear-gradient(120deg,#0f4c81,#1b998b);}
.hero h1 {color: white; margin-bottom: .3rem;}
.card {border: 1px solid #d0d7de; border-radius: 12px; padding: 1rem; min-height: 150px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def _load(key, mtime):
    return joblib.load(MODELS_DIR / f"{key}.joblib")


def load_bundle(key):
    f = MODELS_DIR / f"{key}.joblib"
    if not f.exists():
        return None
    try:
        return _load(key, f.stat().st_mtime)
    except Exception as e:
        st.error(f"Could not load model for {DISEASES[key]['name']}: {e}. "
                 "Please re-run `python train_models.py`.")
        return None


def wkey(key, feat):
    return f"{key}__{feat}"


def nice_step(rng):
    if not rng or rng <= 0:
        return 0.01
    return float(10 ** math.floor(math.log10(rng / 100)))


def fill_example(key, cls):
    """Callback: fill the form with typical values of the disease (1) / healthy (0) class."""
    cfg, bundle = DISEASES[key], load_bundle(key)
    for f in bundle["features"]:
        s = bundle["stats"][f]
        v = s["rep1" if cls else "rep0"]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = s["median"]
        opts = cfg["feature_info"].get(f, {}).get("options")
        if opts:
            inv = {c: l for l, c in opts.items()}
            st.session_state[wkey(key, f)] = inv.get(int(round(v)), next(iter(inv.values())))
        else:
            st.session_state[wkey(key, f)] = int(round(v)) if s["is_int"] else float(v)


def draw_input(key, cfg, bundle, feat, values):
    meta = cfg["feature_info"].get(feat, {})
    s = bundle["stats"][feat]
    label = meta.get("label", feat.replace("_", " ").capitalize())
    help_txt = meta.get("help") or f"Typical range in training data: {s['min']:g} to {s['max']:g}"
    if "options" in meta:
        choice = st.selectbox(label, list(meta["options"]), key=wkey(key, feat), help=meta.get("help"))
        values[feat] = meta["options"][choice]
    elif s["is_int"]:
        values[feat] = st.number_input(label, value=int(round(s["median"])), step=1,
                                       key=wkey(key, feat), help=help_txt)
    else:
        values[feat] = st.number_input(label, value=float(s["median"]),
                                       step=nice_step(s["max"] - s["min"]), format="%g",
                                       key=wkey(key, feat), help=help_txt)


def validate(cfg, bundle, values):
    """Return (errors, warnings). Errors block the prediction."""
    errors, warns = [], []
    zero_ok = set(cfg.get("zero_is_missing", []))
    for f, v in values.items():
        s = bundle["stats"][f]
        label = cfg["feature_info"].get(f, {}).get("label", f)
        if v is None or not np.isfinite(v):
            errors.append(f"'{label}' must be a valid number.")
            continue
        if f in zero_ok and v == 0:
            continue  # 0 means "unknown" for these fields
        if s["min"] >= 0 and v < 0:
            errors.append(f"'{label}' cannot be negative.")
        elif s["max"] > 0 and v > 10 * s["max"]:
            errors.append(f"'{label}' = {v:g} is implausibly large (max in data: {s['max']:g}). "
                          "Check the units.")
        elif (v < s["min"] or v > s["max"]):
            warns.append(f"'{label}' = {v:g} is outside the range seen in training data "
                         f"({s['min']:g} to {s['max']:g}); the prediction may be unreliable.")
    return errors, warns


def missing_model_msg(key):
    st.warning(f"No trained model found for **{DISEASES[key]['name']}**.")
    f = DISEASES[key]["file"]
    if f:
        st.markdown(f"1. Make sure `data/{f}` exists (`python download_datasets.py`).\n"
                    "2. Run `python train_models.py` in the terminal.\n3. Refresh this page.")
    else:
        st.markdown("Run `python train_models.py` in the terminal, then refresh this page.")


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    st.markdown("""<div class="hero"><h1>🩺 General Disease Prediction System</h1>
    <p style="font-size:1.1rem">A machine-learning web app that screens six conditions using
    separate, independently trained models: Diabetes, Heart Disease, Chronic Kidney Disease,
    Liver Disease, Parkinson's Disease and Breast Cancer (tumour classification).</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    st.warning(DISCLAIMER)
    st.subheader("Supported modules")
    keys = list(DISEASES)
    for row in (keys[:3], keys[3:]):
        cols = st.columns(3)
        for col, key in zip(cols, row):
            cfg, b = DISEASES[key], load_bundle(key)
            with col:
                if b:
                    r = b["results"][b["best_model"]]
                    status = (f"✅ Trained · **{b['best_model']}**<br>"
                              f"Test F1 {r['f1']:.2f} · Accuracy {r['accuracy']:.2f}")
                else:
                    status = "⚠️ Not trained yet"
                st.markdown(f"""<div class="card"><h3>{cfg['icon']} {cfg['name']}</h3>
                <small>{cfg['source']}</small><p>{status}</p></div>""", unsafe_allow_html=True)
    st.subheader("How to use")
    st.markdown("1. Open **Disease Prediction** in the sidebar and choose a disease.\n"
                "2. Enter the patient values (or press a *Fill typical values* button).\n"
                "3. Click **Predict** to see the result and its probability.\n"
                "4. Open **Model Performance** to see metrics, confusion matrices and "
                "feature importance for every model.")


def page_predict():
    st.title("🔍 Disease Prediction")
    names = {k: f"{c['icon']} {c['name']}" for k, c in DISEASES.items()}
    key = st.selectbox("Select a disease module", list(DISEASES), format_func=names.get)
    cfg, bundle = DISEASES[key], load_bundle(key)
    st.info(cfg["note"])
    if bundle is None:
        missing_model_msg(key)
        return

    c1, c2, _ = st.columns([1, 1, 3])
    c1.button(f"Fill typical '{cfg['neg_label']}' values", on_click=fill_example, args=(key, 0))
    c2.button(f"Fill typical '{cfg['pos_label']}' values", on_click=fill_example, args=(key, 1))

    values = {}
    feats = bundle["features"]
    if cfg.get("groups"):
        tabs = st.tabs(list(cfg["groups"]))
        for tab, group_feats in zip(tabs, cfg["groups"].values()):
            with tab:
                cols = st.columns(3)
                for i, f in enumerate([g for g in group_feats if g in feats]):
                    with cols[i % 3]:
                        draw_input(key, cfg, bundle, f, values)
    else:
        cols = st.columns(3)
        for i, f in enumerate(feats):
            with cols[i % 3]:
                draw_input(key, cfg, bundle, f, values)

    if st.button("🔮 Predict", type="primary"):
        errors, warns = validate(cfg, bundle, values)
        if errors:
            for e in errors:
                st.error(e)
            return
        for w in warns:
            st.warning(w)
        row = pd.DataFrame([values])[feats].astype(float)
        for f in cfg.get("zero_is_missing", []):
            if f in row and row.loc[0, f] == 0:
                row.loc[0, f] = np.nan
        try:
            p = float(bundle["pipeline"].predict_proba(row)[0][1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
        st.subheader("Result")
        if p >= 0.5:
            st.error(f"**Model output: {cfg['pos_label']}** (estimated probability {p:.1%})")
        else:
            st.success(f"**Model output: {cfg['neg_label']}** (probability of "
                       f"{cfg['pos_label'].lower()}: {p:.1%})")
        st.progress(min(max(p, 0.0), 1.0))
        r = bundle["results"][bundle["best_model"]]
        st.caption(f"Model used: {bundle['best_model']} · trained only on the "
                   f"{cfg['source']} · test accuracy {r['accuracy']:.1%}, recall {r['recall']:.1%}, "
                   f"F1 {r['f1']:.1%}.")
        if key == "breast_cancer":
            st.info("This result classifies a tumour sample as benign or malignant. It is NOT a "
                    "prediction of future cancer risk.")
        st.warning(DISCLAIMER)


def plot_cm(ax, cm, cfg, title):
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=[f"Pred:\n{cfg['neg_label']}", f"Pred:\n{cfg['pos_label']}"],
                yticklabels=[f"Actual:\n{cfg['neg_label']}", f"Actual:\n{cfg['pos_label']}"])
    ax.set_title(title)


def page_performance():
    st.title("📊 Model Performance Dashboard")
    st.caption("All values are computed by train_models.py on a held-out test set that the "
               "models never saw during training or tuning.")
    bundles = {k: load_bundle(k) for k in DISEASES}
    trained = {k: b for k, b in bundles.items() if b}
    if not trained:
        st.warning("No models trained yet. Run `python train_models.py` first.")
        return

    st.subheader("Overview - selected model for each disease")
    rows = []
    for k, b in trained.items():
        r = b["results"][b["best_model"]]
        rows.append({"Disease": b["name"], "Selected model": b["best_model"],
                     **{m.replace("_", "-").title(): round(r[m], 3) for m in METRICS}})
    ov = pd.DataFrame(rows)
    st.dataframe(ov, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=ov.melt(id_vars=["Disease", "Selected model"], var_name="Metric",
                             value_name="Score"), x="Disease", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, fontsize=8, loc="lower center")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.divider()
    key = st.selectbox("Inspect a disease in detail", list(trained),
                       format_func=lambda k: DISEASES[k]["name"])
    b, cfg = trained[key], DISEASES[key]
    m = b["meta"]
    c = st.columns(4)
    c[0].metric("Samples", m["n_rows"])
    c[1].metric("Features", m["n_features"])
    c[2].metric("Train / Test", f"{m['n_train']} / {m['n_test']}")
    c[3].metric("Selected model", b["best_model"])
    st.caption(f"Target: {cfg['target_desc']} · Positive cases: {m['n_pos']}, negative: "
               f"{m['n_neg']} · Split: {m['split']} · Missing cells (imputed): "
               f"{m['missing_cells']} · Trained: {m['trained_at']}")

    st.markdown("#### Model comparison (test set)")
    tbl = pd.DataFrame({n: {**{mm.replace('_', '-').title(): r[mm] for mm in METRICS},
                            "CV F1 (train)": r["cv_f1"]} for n, r in b["results"].items()}).T
    st.dataframe(tbl.round(3))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    long = tbl.drop(columns="CV F1 (train)").reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Score")
    sns.barplot(data=long, x="Metric", y="Score", hue="index", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")
    st.pyplot(fig)

    st.markdown("#### Confusion matrices (test set)")
    cols = st.columns(len(b["results"]))
    for col, (n, r) in zip(cols, b["results"].items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            plot_cm(ax, r["confusion_matrix"], cfg, n)
            st.pyplot(fig)

    st.markdown("#### Feature importance")
    which = st.radio("Model", list(b["results"]), horizontal=True,
                     index=list(b["results"]).index(b["best_model"]))
    imp = pd.Series(b["results"][which]["importance"]).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=imp.values, y=[cfg["feature_info"].get(i, {}).get("label", i) for i in imp.index],
                color="#1b998b", ax=ax)
    ax.set_xlabel("Relative importance (Random Forest: impurity; Logistic Regression: |coefficient|)")
    st.pyplot(fig)
    st.caption(f"Best hyper-parameters found for {which}: {b['results'][which]['best_params']}")


def page_about():
    st.title("ℹ️ About the Project")
    st.markdown("""
**General Disease Prediction System Using Machine Learning** - a third-year IT Engineering
practical (PR12). It demonstrates the complete ML workflow: data loading, cleaning,
train/test splitting, preprocessing pipelines, model comparison, hyper-parameter tuning,
evaluation, model saving and deployment in a web interface.

**Technology:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit

**Method:** For each disease the data is split first; a pipeline (median imputation → scaling →
classifier) is fitted on the training part only; Logistic Regression and Random Forest are tuned
with cross-validated grid search (scoring: F1 of the disease class); the better model by
cross-validation is saved. Test-set metrics are reported once.
""")
    st.subheader("Datasets")
    st.dataframe(pd.DataFrame([{"Module": c["name"], "Dataset": c["source"],
                                "Target": c["target_desc"], "Link": c["page"]}
                               for c in DISEASES.values()]), hide_index=True)
    st.subheader("Important limitations")
    st.markdown("""
- Datasets are small (hundreds of rows) and come from specific populations/hospitals, so the
  models may not generalise to other people.
- Inputs are few clinical numbers; real diagnosis needs far more information and a doctor.
- **Breast cancer** module: classifies an existing tumour sample as *benign vs malignant* from
  lab measurements. It does **not** estimate the risk of developing cancer in the future.
- **Parkinson's** data has several recordings per person; the split is done per person to avoid
  inflated accuracy, but only 31 people exist in the data, so results are uncertain.
""")
    st.warning(DISCLAIMER)


# ----------------------------------------------------------------------------
PAGES = {"🏠 Home": page_home, "🔍 Disease Prediction": page_predict,
         "📊 Model Performance": page_performance, "ℹ️ About Project": page_about}
st.sidebar.title("🩺 Navigation")
choice = st.sidebar.radio("Go to", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(DISCLAIMER)
PAGES[choice]()
