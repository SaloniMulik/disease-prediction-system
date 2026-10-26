"""
train_models.py - trains and evaluates models for all six disease modules.

Usage:
    python train_models.py                 # train everything
    python train_models.py --only heart    # train one module

Method (same for every disease):
  1. Load data, remove duplicate rows.
  2. Split into train / test (stratified; per-person for Parkinson's).
  3. Pipeline = median imputer -> standard scaler -> model. Fitted on TRAIN ONLY,
     so no information from the test set leaks into preprocessing.
  4. Logistic Regression and Random Forest are tuned with GridSearchCV
     (cross-validation inside the training set, scoring = F1 of the disease class).
  5. The model with the better cross-validated F1 is selected (NOT chosen on test data).
  6. Both models are evaluated once on the untouched test set.
  7. Selected pipeline + metrics + plots data are saved to models/<key>.joblib.
All numbers are computed here - nothing is hard-coded.
"""
import argparse
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")
RS = 42


def make_pipeline(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("model", model)])


def model_spaces():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10, 100]}),
        "Random Forest": (
            RandomForestClassifier(random_state=RS, class_weight="balanced"),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [None, 5, 10],
             "model__min_samples_leaf": [1, 3]}),
    }


def split_data(X, y, groups):
    """Return X_tr, X_te, y_tr, y_te, groups_tr, description."""
    if groups is None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RS)
        return X_tr, X_te, y_tr, y_te, None, "80/20 stratified random split"
    for seed in range(RS, RS + 100):  # group split: whole people go to one side only
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=seed).split(X, y, groups))
        if y.iloc[tr].nunique() == 2 and y.iloc[te].value_counts().min() >= 3:
            break
    return (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr],
            "75/25 split by person (no person in both train and test)")


def importance_of(pipe, features):
    m = pipe.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        v = m.feature_importances_
    else:
        v = np.abs(m.coef_[0])  # coefficients are comparable: features are standardised
    v = v / v.sum() if v.sum() > 0 else v
    return {f: float(x) for f, x in zip(features, v)}


def feature_stats(X, y):
    stats = {}
    for c in X.columns:
        col = X[c].dropna()
        s = {"min": float(col.min()), "max": float(col.max()),
             "median": float(col.median()),
             "is_int": bool(np.allclose(col, np.round(col)))}
        for cls in (0, 1):
            sub = X.loc[y == cls, c].dropna()
            if len(sub) == 0:
                s[f"rep{cls}"] = s["median"]
            elif s["is_int"]:
                s[f"rep{cls}"] = float(sub.mode().iloc[0])
            else:
                s[f"rep{cls}"] = float(sub.median())
        stats[c] = s
    return stats


def train_one(key):
    cfg = config.DISEASES[key]
    path = config.dataset_path(key)
    X, y, groups = cfg["loader"](path)
    if y.nunique() != 2 or len(X) < 100:
        raise ValueError(f"Dataset looks wrong: {len(X)} rows, classes={y.unique()}")
    print(f"   rows={len(X)}  features={X.shape[1]}  disease cases={int(y.sum())}  "
          f"healthy={int((y == 0).sum())}  missing cells={int(X.isna().sum().sum())}")

    X_tr, X_te, y_tr, y_te, g_tr, split_desc = split_data(X, y, groups)
    print(f"   split: {split_desc} -> train={len(X_tr)}, test={len(X_te)}")

    if g_tr is not None:
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RS)
        fit_kw = {"groups": g_tr}
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
        fit_kw = {}

    results, fitted = {}, {}
    for name, (est, grid) in model_spaces().items():
        gs = GridSearchCV(make_pipeline(est), grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr, **fit_kw)
        pipe = gs.best_estimator_
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
            "cv_f1": float(gs.best_score_),
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
            "importance": importance_of(pipe, list(X.columns)),
        }
        fitted[name] = pipe
        r = results[name]
        print(f"   {name:<20} CV-F1={r['cv_f1']:.3f} | test: acc={r['accuracy']:.3f} "
              f"prec={r['precision']:.3f} rec={r['recall']:.3f} f1={r['f1']:.3f}")

    best = max(results, key=lambda n: results[n]["cv_f1"])  # ties -> first (simpler) model
    print(f"   -> selected: {best} (highest cross-validated F1 on training data)")

    bundle = {
        "key": key, "name": cfg["name"], "features": list(X.columns),
        "pipeline": fitted[best], "best_model": best, "results": results,
        "stats": feature_stats(X, y),
        "meta": {"n_rows": int(len(X)), "n_features": int(X.shape[1]),
                 "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                 "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                 "n_test_pos": int(y_te.sum()), "split": split_desc,
                 "missing_cells": int(X.isna().sum().sum()),
                 "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "sklearn_version": sklearn.__version__},
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODELS_DIR / f"{key}.joblib")
    return bundle


def write_results(bundles):
    config.DOCS_DIR.mkdir(exist_ok=True)
    rows = []
    lines = ["# Actual Training Results (auto-generated by train_models.py)\n",
             "Metrics are for the positive (disease) class on the held-out test set.\n"]
    for key, b in bundles.items():
        m = b["meta"]
        lines += [f"\n## {b['name']}\n",
                  f"- Rows: {m['n_rows']}, features: {m['n_features']}, "
                  f"disease cases: {m['n_pos']}, non-disease: {m['n_neg']}",
                  f"- Split: {m['split']} (train {m['n_train']}, test {m['n_test']})",
                  f"- Selected model: **{b['best_model']}**\n",
                  "| Model | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in b["results"].items():
            (tn, fp), (fn, tp) = r["confusion_matrix"]
            lines.append(f"| {n} | {r['cv_f1']:.3f} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
                         f"{r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {tn} | {fp} | {fn} | {tp} |")
            rows.append({"disease": b["name"], "model": n,
                         "selected": n == b["best_model"], **{k: round(r[k], 4) for k in
                         ["cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc"]},
                         "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    (config.DOCS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(config.MODELS_DIR / "training_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(config.DISEASES), help="train a single module")
    args = ap.parse_args()
    keys = [args.only] if args.only else list(config.DISEASES)

    status, bundles = {}, {}
    for key in keys:
        cfg = config.DISEASES[key]
        print(f"\n=== {cfg['name']} ===")
        path = config.dataset_path(key)
        if path is not None and not path.exists():
            print(f"   MISSING DATASET: {path}\n   Run 'python download_datasets.py' "
                  f"or see README. Info page: {cfg['page']}")
            status[key] = "MISSING DATASET"
            continue
        try:
            bundles[key] = train_one(key)
            status[key] = "trained"
        except Exception as e:  # keep going with the other diseases
            print(f"   FAILED: {type(e).__name__}: {e}")
            status[key] = f"FAILED ({e})"

    print("\n================ SUMMARY ================")
    for key in keys:
        print(f"{config.DISEASES[key]['name']:<26} {status[key]}")
    if bundles:
        # merge with previously trained modules so RESULTS.md always lists everything
        for key in config.DISEASES:
            f = config.MODELS_DIR / f"{key}.joblib"
            if key not in bundles and f.exists():
                bundles[key] = joblib.load(f)
        write_results({k: bundles[k] for k in config.DISEASES if k in bundles})
        print("\nSaved models in 'models/' and results table in 'docs/RESULTS.md'.")
        print("Next step:  streamlit run app.py")
    else:
        print("\nNo model was trained. Fix the problems above and run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
