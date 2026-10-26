# General Disease Prediction System Using Machine Learning
**PR12 – AI/ML Practical · Third-Year IT Engineering**

*(Add your name, roll number, guide and college on the title page. Paste your real results into Section 11 from `docs/RESULTS.md` after running `train_models.py`.)*

## 1. Abstract
Early screening of chronic diseases can reduce complications, but clinical decision support is often unavailable in resource-limited settings. This project develops a web-based General Disease Prediction System that applies supervised machine learning to six publicly available medical datasets: diabetes, heart disease, chronic kidney disease, liver disease, Parkinson's disease and breast cancer (benign/malignant tumour classification). For each disease a separate pipeline of imputation, scaling and a classifier (Logistic Regression or Random Forest, tuned by cross-validated grid search) is trained and evaluated on a held-out test set using accuracy, precision, recall, F1-score and confusion matrices. The best model per disease is saved and served through a Streamlit application with input validation, a performance dashboard and feature-importance charts. The system is an educational prototype and not a diagnostic tool.

## 2. Problem Statement
Medical test values are numerous and hard to interpret together. There is a need for a simple system that learns patterns from historical patient data and gives a quick, explainable risk indication for multiple diseases from one interface, while being honest about its limitations.

## 3. Objectives
1. Build separate, correctly pre-processed ML models for six diseases using public datasets.
2. Compare Logistic Regression and Random Forest using standard classification metrics.
3. Prevent data leakage through proper splitting, pipelines and cross-validation.
4. Deploy the models in a user-friendly Streamlit web application with validation and visual analytics.
5. Document limitations, ethical considerations and responsible use.

## 4. Literature Survey Suggestions
*(Verify each reference's exact details in Google Scholar / IEEE Xplore before citing.)*
- Smith et al. (1988), ADAP learning algorithm for forecasting diabetes onset – origin of the Pima dataset.
- Detrano et al. (1989), *International application of a new probability algorithm for the diagnosis of coronary artery disease*, Am. J. Cardiology – Cleveland heart dataset.
- Wolberg & Mangasarian (1990), breast-cytology diagnosis by multisurface pattern separation, PNAS – Wisconsin breast cancer data.
- Little et al. (2007), voice-disorder detection using nonlinear recurrence and fractal scaling, BioMedical Engineering OnLine – Parkinson's voice dataset.
- Breiman (2001), *Random Forests*, Machine Learning 45(1).
- Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*, JMLR 12.
- Search terms: "machine learning disease prediction review", "chronic kidney disease classification UCI", "data leakage in medical machine learning", "multiple disease prediction system Streamlit".

## 5. Existing and Proposed Systems
**Existing:** (a) Manual interpretation of reports by doctors – accurate but time-consuming and not always available. (b) Single-disease research models – usually notebooks with no user interface, one disease each. (c) Commercial symptom checkers – closed, often paid, not transparent.

**Proposed:** one open-source application hosting six independent models with a uniform workflow, transparent metrics, feature-importance explanations and input validation, built only with free tools.

## 6. Methodology
1. **Data collection** – five UCI/GitHub files plus scikit-learn's built-in Wisconsin dataset.
2. **Cleaning** – duplicates removed; impossible zeros in Pima Glucose, BloodPressure, SkinThickness, Insulin and BMI treated as missing; CKD "?" treated as missing; categorical text coded numerically (e.g. normal=0, abnormal=1).
3. **Splitting** – stratified 80/20 split (75/25 grouped by person for Parkinson's).
4. **Preprocessing pipeline** – median imputer → StandardScaler → classifier, fitted on training data only.
5. **Training and tuning** – GridSearchCV (5-fold stratified; 4-fold stratified-group for Parkinson's), scoring = F1 of the disease class; `class_weight="balanced"` handles imbalance.
6. **Model selection** – highest cross-validated F1 on training data (test set not used for selection).
7. **Evaluation** – accuracy, precision, recall, F1, ROC-AUC and confusion matrix on the test set.
8. **Deployment** – pipelines saved with joblib and loaded by the Streamlit app.

## 7. System Architecture
```
 Public datasets ──► config.py loaders ──► train_models.py
                                              │  split → Pipeline(imputer→scaler→model) → GridSearchCV
                                              ▼
                                   models/<disease>.joblib  (pipeline + metrics + feature stats)
                                              │
 User ◄──► Streamlit app.py:  Home │ Prediction form │ Performance dashboard │ About
                              input validation → pipeline.predict_proba → result + disclaimer
```
Each disease has its own file, feature list and pipeline; inputs are never shared between modules.

## 8. Dataset Descriptions
*(Sizes from the dataset documentation; the training log prints the exact counts after duplicate removal.)*

| Disease | Source | Approx. size | Features | Target |
|---|---|---|---|---|
| Diabetes | Pima Indians (NIDDK/UCI) | 768 × 8 | pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, pedigree function, age | Outcome (1 = diabetic) |
| Heart | Cleveland (UCI) | 303 × 13 | age, sex, chest-pain type, BP, cholesterol, fasting sugar, ECG, max heart rate, exercise angina, ST depression, slope, vessels, thal | num>0 = disease |
| CKD | UCI (Apollo hospital) | 400 × 24 | blood/urine tests (sg, albumin, sugar, creatinine, urea, haemoglobin…) and clinical findings (htn, dm, appetite…) | class = ckd |
| Liver | ILPD (UCI) | 583 × 10 | age, gender, bilirubin, enzymes (ALP, ALT, AST), proteins, albumin, A/G ratio | Selector = 1 liver patient |
| Parkinson's | UCI (Little et al.) | 195 recordings of 31 people × 22 | jitter, shimmer, HNR, RPDE, DFA, spread1/2, PPE… | status = 1 |
| Breast cancer | Wisconsin Diagnostic | 569 × 30 | mean, standard error and worst of 10 nucleus measurements | malignant (1) / benign (0) |

*Breast cancer note:* the target is the diagnosis of a tumour sample already examined by fine-needle aspiration. It is **not** a prediction of whether a healthy person will develop cancer.

## 9. Algorithms Used
- **Logistic Regression** – linear model that outputs a probability through the sigmoid function; regularisation strength C tuned in {0.01, 0.1, 1, 10, 100}. Fast, interpretable, strong on roughly linear data.
- **Random Forest** – ensemble of decision trees on bootstrap samples with random feature subsets; prediction by voting. Tuned: n_estimators {100, 300}, max_depth {None, 5, 10}, min_samples_leaf {1, 3}. Handles non-linearity and gives feature importance.
- **Supporting methods:** median imputation, standardisation, stratified K-fold, grouped K-fold, grid search.

## 10. Model Evaluation
Positive class = disease. TP/FP/TN/FN from the confusion matrix.
- Accuracy = (TP+TN)/all · Precision = TP/(TP+FP) · Recall = TP/(TP+FN) · F1 = 2PR/(P+R) · ROC-AUC = ranking quality.
- In screening, **recall** (not missing sick patients) is especially important, so F1 of the disease class is used for tuning and selection rather than plain accuracy.
- Test sets are small (e.g. ~40–120 rows), so one extra misclassified patient changes the scores noticeably.

## 11. Actual Results and Discussion
**Results table:** copy the per-disease tables from `docs/RESULTS.md` (generated by `train_models.py` from your own run) and screenshots of the confusion matrices from the *Model Performance* page. Do not type numbers from anywhere else.

**What to discuss (check each point against your own numbers):**
1. Which model was selected for each disease and by how much it beat the other.
2. Compare recall vs precision: for which disease does the model miss more sick patients than it falsely alarms?
3. CKD is usually a very easy dataset (strongly separable lab values); near-perfect scores should be discussed with caution given only ~400 patients and may not hold in practice.
4. Diabetes and Liver are typically the hardest of the six (overlapping classes, few features); lower scores are expected, not a bug.
5. Parkinson's: explain that a person-wise split gives lower but more honest scores than a random split on recordings.
6. Use the feature-importance chart: e.g. do glucose/BMI (diabetes), creatinine/haemoglobin/specific gravity (CKD), or worst-radius/concave-points (breast cancer) rank high? Does this match medical knowledge?
7. Note the gap between cross-validated F1 and test F1 as a sign of over- or under-fitting.

## 12. Test Cases
| # | Test | Input / action | Expected behaviour | Pass/Fail (tick after running) |
|---|---|---|---|---|
| 1 | Training runs | `python train_models.py` with all datasets | Six modules print metrics; `models/*.joblib` and `docs/RESULTS.md` created | ☐ |
| 2 | Missing dataset | Rename `data/parkinsons.data`, retrain | "MISSING DATASET" for Parkinson's; others still train | ☐ |
| 3 | Smoke test | `python smoke_test.py` | All PASS, probabilities within 0–1 | ☐ |
| 4 | Home page | Open app | Six cards with ✅ trained status, disclaimer visible | ☐ |
| 5 | Typical healthy profile | Prediction page → "Fill typical healthy values" → Predict | Result shown with probability, usually the non-disease label | ☐ |
| 6 | Typical disease profile | Press "Fill typical disease values" → Predict | Result shown, usually the disease label | ☐ |
| 7 | Negative age | Enter age = −5 → Predict | Red error "cannot be negative"; no prediction | ☐ |
| 8 | Absurd value | BMI = 99999 | Error "implausibly large… check units" | ☐ |
| 9 | Out-of-range value | Glucose = 400 | Yellow warning that value is outside the training range; prediction still shown | ☐ |
| 10 | Unknown test | Diabetes Insulin = 0 | Treated as missing and imputed; no error | ☐ |
| 11 | Module isolation | Switch disease in dropdown | Different input fields and a different model each time | ☐ |
| 12 | Dashboard | Model Performance page | Overview chart, comparison table, 2 confusion matrices, feature importance | ☐ |
| 13 | No models | Delete `models/*.joblib` | Friendly "not trained" instructions, no crash | ☐ |
| 14 | Breast-cancer note | Predict on breast cancer | Notice that result is benign/malignant, not future risk | ☐ |

## 13. Conclusion
The project demonstrates a complete, leakage-aware machine-learning workflow for six medical classification tasks and delivers it through an accessible web interface. Model comparison on real test metrics shows how algorithm choice, preprocessing and data quality affect screening performance. The system is intended only for learning.

## 14. Limitations and Future Scope
**Limitations:** small, old, single-source datasets; limited demographic diversity (e.g. Pima women only); few features; binary outputs without severity; Parkinson's data has only 31 subjects; Cleveland heart and CKD data contain missing values; no external validation; models are not calibrated for clinical use; ordinal coding of some categories (chest pain, thal) is a simplification.

**Future scope:** larger multi-centre datasets and external validation; probability calibration; SHAP/LIME explanations; more models (gradient boosting, SVM, neural networks); more diseases; PDF report export; user accounts and secure storage with consent; deployment on Streamlit Community Cloud; clinician review of outputs.
