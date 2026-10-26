# Viva Questions and Simple Answers

**1. What is your project about?**
A web app that predicts six conditions (diabetes, heart disease, kidney disease, liver disease, Parkinson's, breast tumour type) using a separate machine-learning model for each.

**2. Is it a real diagnostic tool?**
No. It is an educational prototype trained on small public datasets. Only a doctor can diagnose.

**3. What type of ML problem is this?**
Supervised binary classification – the label is disease (1) or no disease (0).

**4. Why a separate model for each disease?**
Each disease has different inputs, data and patterns. Mixing them would make no sense.

**5. Which algorithms did you use and why?**
Logistic Regression (simple, interpretable baseline) and Random Forest (handles non-linear patterns, gives feature importance).

**6. How does Logistic Regression work?**
It computes a weighted sum of features and passes it through a sigmoid to get a probability between 0 and 1.

**7. How does Random Forest work?**
It trains many decision trees on random samples/features and combines their votes, which reduces overfitting.

**8. What is a train-test split and why is it needed?**
Data is divided so the model learns on the training part and is evaluated on unseen test data, giving an honest performance estimate.

**9. What is data leakage? How did you prevent it?**
Leakage is when test-set information influences training. I split first, kept imputation and scaling inside a Pipeline fitted only on training data, tuned using cross-validation inside the training set, and removed duplicate rows before splitting.

**10. Why was Parkinson's split by person?**
The dataset has about 6 recordings per person. A random split would put the same person's voice in both train and test, inflating accuracy. Grouped splitting keeps each person on one side only.

**11. What is a Pipeline in scikit-learn?**
A chain of steps (imputer → scaler → model) applied in order, so the same preprocessing is used in training and in the app.

**12. How did you handle missing values?**
Median imputation. In the Pima data, zeros in glucose, blood pressure, skin thickness, insulin and BMI are impossible, so they were treated as missing.

**13. Why scale the features?**
Logistic Regression is sensitive to feature scale; scaling puts all features on a comparable range.

**14. Explain the confusion matrix.**
A table of true negatives, false positives, false negatives and true positives showing exactly which errors the model makes.

**15. Define accuracy, precision, recall and F1.**
Accuracy = correct/total. Precision = of predicted-sick, how many truly are. Recall = of truly sick, how many were found. F1 = harmonic mean of precision and recall.

**16. Why is accuracy alone not enough?**
With imbalanced classes a model can score high accuracy by predicting the majority class. Recall and F1 show how well the disease class is found.

**17. Which metric matters most for disease screening?**
Recall, because missing a sick patient (false negative) is worse than a false alarm; so I tune using F1 of the disease class.

**18. What is cross-validation?**
The training data is split into K parts; the model trains on K−1 and validates on the remaining one, rotating K times, to get a stable score for tuning.

**19. What is GridSearchCV?**
It tries every combination of chosen hyper-parameters with cross-validation and keeps the best combination.

**20. What is a hyper-parameter? Give examples.**
A setting chosen before training, e.g. C in Logistic Regression and n_estimators, max_depth in Random Forest.

**21. How did you choose the final model for each disease?**
By the best cross-validated F1 on the training data – not by looking at the test set.

**22. What is class imbalance and how was it handled?**
When one class has far fewer samples. I used `class_weight="balanced"` and stratified splits so both classes appear in train and test.

**23. What is overfitting? How can you detect it?**
The model memorises training data and fails on new data. A large gap between training/CV score and test score indicates it.

**24. What does feature importance show?**
Which inputs influence the prediction most (Random Forest impurity importance; for Logistic Regression, absolute coefficient size). It shows association, not medical causation.

**25. Why is breast cancer here "classification" and not "risk prediction"?**
The dataset has measurements of a tumour sample and labels it benign or malignant. It cannot tell whether a healthy person will get cancer in future.

**26. Why does the CKD model often score almost 100%?**
CKD lab values are strongly different between patients and non-patients and the dataset is small; such a score may not hold on real hospital data.

**27. How is the model saved and used in the app?**
`joblib.dump` saves the whole pipeline; the Streamlit app loads it with `joblib.load` and calls `predict_proba`.

**28. Why Streamlit?**
It builds interactive web apps in pure Python without HTML/JavaScript, ideal for ML demos.

**29. What input validation did you add?**
Negative values are rejected for non-negative features, absurdly large values are rejected, out-of-range values give a warning, and invalid numbers show errors.

**30. What are the main limitations and future improvements?**
Small datasets from specific populations, few features, no external validation. Future work: bigger datasets, SHAP explanations, calibration, more models and diseases.

**31. What are the ethical concerns?**
Wrong predictions can cause harm or false reassurance, patient data is sensitive, and models can be biased by the population in the training data – hence the disclaimer.
