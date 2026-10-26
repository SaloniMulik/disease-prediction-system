# General Disease Prediction System Using Machine Learning (PR12)

A Streamlit web app with six independent ML modules: **Diabetes, Heart Disease, Chronic Kidney
Disease, Liver Disease, Parkinson's Disease, Breast Cancer (benign/malignant)**.

> ⚠️ Educational prototype - **not** a medical diagnostic tool.

## Honest status of this delivery

| Item | Status |
|---|---|
| Training code (all 6 modules) | Run and verified in the author's sandbox. Breast cancer was run on its real bundled dataset. The other five were run only on throw-away synthetic files in the same format, to test the code paths. |
| Real accuracy numbers for 5 datasets | **Not available yet** - the sandbox had no internet. You will get the real numbers when you run `train_models.py`; they are written to `docs/RESULTS.md`. |
| Dataset download links | Official UCI / GitHub links, **not tested live**. If one fails, the script prints the manual steps. |
| Streamlit app | All 4 pages' Python logic was run with a stubbed Streamlit and the file compiles. The app was **not opened in a real browser** (Streamlit couldn't be installed offline). If you see a Streamlit error, send me the message. |

## Fastest way (Windows) - one double-click

1. Install **Python 3.10 - 3.12** from python.org (tick **"Add python.exe to PATH"**).
2. Unzip the project, e.g. to `C:\Users\<you>\Desktop\disease_prediction`.
3. Double-click **`run_all.bat`**. It does steps 3-7 below automatically and opens the app.

## Step-by-step (VS Code)

**1. Open the project in VS Code** - right-click the unzipped folder → *Open with Code*, or in
Command Prompt:
```
cd %USERPROFILE%\Desktop\disease_prediction
code .
```
**2. Open a terminal inside VS Code:** menu *Terminal → New Terminal*. Click the small ▾ next to
`+` and choose **Command Prompt** (avoids PowerShell permission errors). All commands below are
typed here, in the project folder.

**3. Create and activate the virtual environment**
```
python -m venv .venv
.venv\Scripts\activate
```
Expected: the prompt now starts with `(.venv)`. If VS Code asks to use the new environment, click *Yes*.
(In PowerShell, if activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first.)

**4. Install dependencies** (1-3 minutes)
```
pip install -r requirements.txt
```
Expected: ends with `Successfully installed ...`

**5. Download the datasets** (puts 5 files into `data/`)
```
python download_datasets.py
```
Expected: five `[ ok ]` lines and `5/5 datasets ready`. Any `[MANUAL]` line tells you which page to
open and what file name to save in `data/` (see table below).

**6. Train all six models** (about 1-3 minutes)
```
python train_models.py
```
Expected: for each disease it prints rows/features, train/test sizes, and CV-F1 plus test accuracy,
precision, recall, F1 for both models, then a SUMMARY. A missing dataset shows `MISSING DATASET`
and the other modules still train. Results table is written to `docs/RESULTS.md`.
Optional check: `python smoke_test.py` (expects `All checks passed.`)

**7. Start the app and open it in the browser**
```
streamlit run app.py
```
Expected: the browser opens `http://localhost:8501` automatically (if not, paste that address into
Chrome/Edge). If Streamlit asks for an email, just press **Enter**. Stop the app with **Ctrl + C**.

## Datasets

| Module | Exact file in `data/` | Direct link | Info page |
|---|---|---|---|
| Diabetes | `diabetes.csv` | https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv | https://archive.ics.uci.edu/ml/datasets/diabetes |
| Heart | `heart_cleveland.data` | https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Kidney | `chronic_kidney_disease.arff` | https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip (take the `.arff` inside) | https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease |
| Liver | `ilpd_liver.csv` | https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv | https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset |
| Parkinson's | `parkinsons.data` | https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data | https://archive.ics.uci.edu/dataset/174/parkinsons |
| Breast cancer | *(none - built into scikit-learn)* | - | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |

Manual download works too: save the file with the exact name above into `data/`. The loaders also
accept the common Kaggle copies of the diabetes and liver CSVs (with a header row).

Targets (1 = positive class): diabetes `Outcome`; heart `num>0`; CKD `class=ckd`; liver
`Selector=1`; Parkinson's `status=1`; breast cancer = malignant.

## Project structure
```
app.py                 Streamlit app (Home, Prediction, Performance, About)
train_models.py        Trains + evaluates + saves all six models
config.py              Dataset loaders and form definitions for every disease
download_datasets.py   Downloads the five datasets
smoke_test.py          Verifies saved models load and predict
run_all.bat            One-click Windows setup
data/  models/         Datasets in, trained .joblib files out
docs/REPORT.md         College report      docs/VIVA.md   Viva Q&A
docs/RESULTS.md        Created by train_models.py with your real results
```

## ML design (short)
- **No leakage:** split first; imputation + scaling live inside a scikit-learn `Pipeline` fitted on
  training data only; tuning uses cross-validation inside the training set; the model is chosen by CV
  score, and the test set is used once for reporting. Duplicate rows are removed before splitting.
- **Parkinson's:** the data has ~6 recordings per person, so a random split would leak voices into the
  test set. The split and CV are done **by person**.
- **Models:** Logistic Regression and Random Forest (`class_weight="balanced"`), tuned by `GridSearchCV`, scoring F1 of the disease class.
- **Missing values:** zeros in Pima Glucose/BP/Skin/Insulin/BMI are treated as missing; CKD `?` values are missing; all are median-imputed.
- **Breast cancer** is benign-vs-malignant classification of an existing tumour sample, **not** future-risk prediction.

## Troubleshooting
- `python` not recognised → reinstall Python with *Add to PATH*, reopen VS Code.
- `streamlit` not recognised → the venv isn't active; run `.venv\Scripts\activate`.
- Model warnings about scikit-learn versions → re-run `python train_models.py` on your machine.
- Port busy → `streamlit run app.py --server.port 8502`.
