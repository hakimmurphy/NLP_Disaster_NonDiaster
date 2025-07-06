# NLP_Disaster_NonDiaster
This repository contains a complete end‑to‑end pipeline for detecting disaster‑reladd tweets. Starting from raw, labeled Twitter data, we perform exploratory data analysis (EDA), text preprocessing, feature engineering, model training, and evaluation, culminating in a reliable classifier based on TF–IDF and Logistic Regression.

📝 Project Overview

**Objective**: Build an automated system to classify tweets as disaster-related (1) or non-disaster (0), enabling emergency responders and analysts to filter relevant information quickly.

**Key Steps**:

1. **Exploratory Data Analysis (EDA)**

- Examined class balance (56% vs. 44%).

- Analyzed tweet length distributions and word-frequency histograms.

- Visualized top tokens and word clouds to identify high-signal vocabulary.

2. **Preprocessing & Feature Engineering**

- Cleaned text: lowercasing, punctuation removal, stop-word filtering, slang normalization.

- Generated TF–IDF features with both word- and character-level n-grams.

- Tuned TF–IDF parameters: sublinear_tf=True, max_df=0.8, min_df=3, max_features=10000.

3. **Model Training & Tuning**

- Evaluated multiple classifiers: Logistic Regression, SVM, Random Forest, k-NN.

- Used stratified 5-fold cross-validation optimizing for F1-score.

- Addressed class imbalance with class_weight='balanced' and decision-threshold calibration.

4. **Evaluation & Results**

- Best model: TF–IDF + Logistic Regression (C=1.0)

- Cross-validated F1-score: ~0.81

- Generated confusion matrix, precision/recall breakdown, and ROC curve.

📊 **Results Summary**

- Best Pipeline: TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_df=0.8, min_df=3, max_features=10000) + LogisticRegression(C=1.0, class_weight='balanced')

- F1-Score: 0.81 (cross-validated)

- Precision / Recall: ~0.80 / ~0.82

🔍**Next Steps**

-** Ensemble models**: Combine top pipelines via soft voting to improve robustness.

- **Transformers: Integrate** sentence-transformers or fine-tune BERT for richer embeddings.

- **Meta-features**: Add hashtag/mention counts, sentiment scores, or topic distributions.

- **Threshold tuning**: Optimize the decision threshold for higher recall in critical scenarios.

👥 **Contributors**

**Hakim Murphy** – Data analysis and modeling


