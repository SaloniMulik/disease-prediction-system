"""
config.py - one place that describes the five disease modules:
  * where the dataset file is and how to load it
  * what the target means (1 = disease present / malignant, 0 = not)
  * how each input field should look in the Streamlit form

Every loader returns (X, y, groups):
  X      - DataFrame of purely numeric features (categories are coded 0/1/...)
  y      - Series, 1 = disease (positive class), 0 = no disease
  groups - Series of subject IDs (only for Parkinson's) or None
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
DOCS_DIR = BASE_DIR / "docs"


def _starts_numeric(path):
    """True if the first cell of the file is a number (i.e. file has no header)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().split(",")[0].strip()
    try:
        float(first)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_diabetes(path):
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(path, header=None, names=cols) if _starts_numeric(path) \
        else pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    # In this dataset a value of 0 for these columns is physically impossible -> missing.
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[c] = df[c].replace(0, np.nan)
    return df.drop(columns="Outcome"), df["Outcome"].astype(int), None


def load_heart(path):
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = pd.read_csv(path, header=None, names=cols, na_values="?")
    df = df.drop_duplicates().reset_index(drop=True)
    y = (df["num"] > 0).astype(int)  # 0 = no disease, 1-4 = disease present
    return df.drop(columns="num"), y, None


def load_liver(path):
    cols = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
            "Albumin_Globulin_Ratio", "Selector"]
    if _starts_numeric(path):
        df = pd.read_csv(path, header=None, names=cols)
    else:  # e.g. the Kaggle copy that has a header row; use position, not names
        df = pd.read_csv(path)
        df.columns = cols
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map(
        {"male": 1, "female": 0})
    df = df.drop_duplicates().reset_index(drop=True)  # ILPD contains duplicate rows
    y = (df["Selector"] == 1).astype(int)  # 1 = liver patient, 2 = not
    return df.drop(columns="Selector"), y, None


def load_parkinsons(path):
    df = pd.read_csv(path)
    groups = df["name"].str.rsplit("_", n=1).str[0]  # e.g. phon_R01_S01 = one person
    y = df["status"].astype(int)
    X = df.drop(columns=["name", "status"])
    return X, y, groups


def load_breast_cancer_data(path=None):
    """Built into scikit-learn (Wisconsin Diagnostic) - no download needed."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    y = (d.target == 0).astype(int)  # sklearn: 0 = malignant, 1 = benign -> 1 = malignant
    return d.data.copy(), y, None


# --------------------------------------------------------------------------
# UI metadata (labels, help text, categorical options)
# --------------------------------------------------------------------------
YES_NO = {"No": 0, "Yes": 1}
NORMAL_ABN = {"Normal": 0, "Abnormal": 1}
PRESENT = {"Not present": 0, "Present": 1}

_bc_base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal dimension"]

DISEASES = {
    "diabetes": dict(
        name="Diabetes", icon="🩸", loader=load_diabetes, file="diabetes.csv",
        pos_label="Diabetic", neg_label="Non-diabetic",
        source="Pima Indians Diabetes Database (UCI / NIDDK)",
        page="https://archive.ics.uci.edu/ml/datasets/diabetes",
        target_desc="Outcome: 1 = diabetes within 5 years, 0 = no diabetes",
        zero_is_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        note="Dataset: women of Pima Indian heritage aged 21+. Enter 0 for a test you "
             "do not have (Glucose, BloodPressure, SkinThickness, Insulin, BMI) - it "
             "will be treated as 'unknown' and filled in automatically.",
        feature_info={
            "Pregnancies": {"label": "Number of pregnancies"},
            "Glucose": {"label": "Plasma glucose (mg/dL)", "help": "2-hour oral glucose tolerance test"},
            "BloodPressure": {"label": "Diastolic blood pressure (mm Hg)"},
            "SkinThickness": {"label": "Triceps skin-fold thickness (mm)"},
            "Insulin": {"label": "2-hour serum insulin (mu U/ml)"},
            "BMI": {"label": "Body mass index (kg/m²)"},
            "DiabetesPedigreeFunction": {"label": "Diabetes pedigree function", "help": "Family-history score"},
            "Age": {"label": "Age (years)"},
        }),
    "heart": dict(
        name="Heart Disease", icon="❤️", loader=load_heart, file="heart_cleveland.data",
        pos_label="Heart disease", neg_label="No heart disease",
        source="Heart Disease Data Set - Cleveland subset (UCI)",
        page="https://archive.ics.uci.edu/ml/datasets/heart+disease",
        target_desc="num: 0 = no disease, 1-4 = disease present (converted to 1)",
        note="Dataset: 303 patients from the Cleveland Clinic.",
        feature_info={
            "age": {"label": "Age (years)"},
            "sex": {"label": "Sex", "options": {"Female": 0, "Male": 1}},
            "cp": {"label": "Chest pain type", "options": {
                "Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}},
            "trestbps": {"label": "Resting blood pressure (mm Hg)"},
            "chol": {"label": "Serum cholesterol (mg/dL)"},
            "fbs": {"label": "Fasting blood sugar > 120 mg/dL", "options": {"No": 0, "Yes": 1}},
            "restecg": {"label": "Resting ECG result", "options": {
                "Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}},
            "thalach": {"label": "Maximum heart rate achieved (bpm)"},
            "exang": {"label": "Exercise-induced angina", "options": YES_NO},
            "oldpeak": {"label": "ST depression (exercise vs rest)"},
            "slope": {"label": "Slope of peak-exercise ST segment", "options": {
                "Upsloping": 1, "Flat": 2, "Downsloping": 3}},
            "ca": {"label": "Major vessels coloured by fluoroscopy", "options": {
                "0": 0, "1": 1, "2": 2, "3": 3}},
            "thal": {"label": "Thalassemia (thallium stress test)", "options": {
                "Normal": 3, "Fixed defect": 6, "Reversible defect": 7}},
        }),
    "liver": dict(
        name="Liver Disease", icon="🧪", loader=load_liver, file="ilpd_liver.csv",
        pos_label="Liver disease", neg_label="No liver disease",
        source="Indian Liver Patient Dataset - ILPD (UCI)",
        page="https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
        target_desc="Selector: 1 = liver patient, 2 = non-patient (converted to 1 / 0)",
        note="Dataset: 583 patients from North-East Andhra Pradesh, India.",
        feature_info={
            "Age": {"label": "Age (years)"},
            "Gender": {"label": "Gender", "options": {"Female": 0, "Male": 1}},
            "Total_Bilirubin": {"label": "Total bilirubin (mg/dL)"},
            "Direct_Bilirubin": {"label": "Direct bilirubin (mg/dL)"},
            "Alkaline_Phosphotase": {"label": "Alkaline phosphatase (IU/L)"},
            "Alamine_Aminotransferase": {"label": "ALT / SGPT (IU/L)"},
            "Aspartate_Aminotransferase": {"label": "AST / SGOT (IU/L)"},
            "Total_Proteins": {"label": "Total proteins (g/dL)"},
            "Albumin": {"label": "Albumin (g/dL)"},
            "Albumin_Globulin_Ratio": {"label": "Albumin / globulin ratio"},
        }),
    "parkinsons": dict(
        name="Parkinson's Disease", icon="🧠", loader=load_parkinsons,
        file="parkinsons.data",
        pos_label="Parkinson's", neg_label="Healthy",
        source="Parkinsons Data Set (UCI, Little et al.)",
        page="https://archive.ics.uci.edu/dataset/174/parkinsons",
        target_desc="status: 1 = Parkinson's, 0 = healthy",
        note="Features are acoustic measurements from sustained-vowel voice recordings "
             "(195 recordings of 31 people). They cannot be typed from memory - use the "
             "'Fill typical values' buttons for a demo. Train/test split is done per PERSON "
             "so the same voice never appears in both sets.",
        feature_info={},
    ),
    "breast_cancer": dict(
        name="Breast Cancer", icon="🎗️", loader=load_breast_cancer_data, file=None,
        pos_label="Malignant", neg_label="Benign",
        source="Breast Cancer Wisconsin (Diagnostic) - bundled with scikit-learn",
        page="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        target_desc="diagnosis: malignant (1) / benign (0) for an already-detected tumour",
        note="IMPORTANT: this classifies a tumour sample as benign vs malignant from "
             "cell-nucleus measurements of a fine-needle aspirate. It does NOT predict "
             "whether a healthy person will develop cancer in the future.",
        groups={"Mean values": [f"mean {b}" for b in _bc_base],
                "Standard error": [f"{b} error" for b in _bc_base],
                "Worst values": [f"worst {b}" for b in _bc_base]},
        feature_info={},
    ),
}


def dataset_path(key):
    f = DISEASES[key]["file"]
    return None if f is None else DATA_DIR / f
