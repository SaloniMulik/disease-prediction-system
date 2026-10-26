"""
smoke_test.py - quick check that every trained model loads and predicts.
Run after training:  python smoke_test.py
"""
import joblib
import pandas as pd

import config

ok = True
for key, cfg in config.DISEASES.items():
    f = config.MODELS_DIR / f"{key}.joblib"
    if not f.exists():
        print(f"[SKIP] {cfg['name']}: model file not found")
        continue
    b = joblib.load(f)
    row = pd.DataFrame([{k: v["median"] for k, v in b["stats"].items()}])[b["features"]]
    for cls in (0, 1):  # typical healthy / typical disease profile
        rep = pd.DataFrame([{k: v[f"rep{cls}"] for k, v in b["stats"].items()}])[b["features"]]
        p = b["pipeline"].predict_proba(rep)[0][1]
        good = 0.0 <= p <= 1.0
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {cfg['name']:<24} typical class {cls}: "
              f"P(disease)={p:.3f}")
print("\nAll checks passed." if ok else "\nSome checks FAILED.")
