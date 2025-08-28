# Fake or Real: The Impostor Hunt (Kaggle)

Kaggle competition: [Competition page](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt)  
Goal: For each test sample (pair of two article variants) predict which file (`file_1` or `file_2`) is the authentic (real) text.

## Repository Structure

- `data/`
  - `train/` raw training article directories
  - `test/` raw test article directories
  - `train.csv` (mapping: id → real_text_id)
  - `traditional.csv` (engineered training dataset)
  - `train_processed.csv`, `test_processed.csv` (intermediate processed data)
  - `submissions/` stored model submission files
- `notebooks/`
  - `create_dataset.ipynb` – builds processed train/test datasets
  - `Tradition_ML.ipynb` – traditional feature engineering + baseline modeling
  - `Modeling.ipynb` – TF‑IDF & embedding pipelines, model training & inference

## Problem Framing

Each training row corresponds to two textual variants (`file_1`, `file_2`) plus a label `real_text_id` ∈ {1,2}.  
Approach: Convert pairwise problem into standard binary classification by:

1. Splitting each row into two records (text + target: real/fake) using `real_text_id`.
2. Concatenating into a single supervised dataset (`text`, `target`).

See data assembly in [notebooks/Tradition_ML.ipynb](notebooks/Tradition_ML.ipynb).

## Data Preparation

Implemented in [notebooks/create_dataset.ipynb](notebooks/create_dataset.ipynb) and [notebooks/Tradition_ML.ipynb](notebooks/Tradition_ML.ipynb):

- Load raw mapping (`train.csv`).
- Join with extracted text contents for `file_1` and `file_2`.
- Reshape to long format: one row per (text, target).
- Drop/inspect missing values (none after processing).
- Export to `traditional.csv` and `train_processed.csv` / `test_processed.csv`.

## Feature Engineering

From [notebooks/Tradition_ML.ipynb](notebooks/Tradition_ML.ipynb) & [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb):

- Basic lexical features:
  - `text_length` = character count
  - `avg_word_length` = mean token length
- TF‑IDF vectorization (unigram default; extendable).
- Sentence embeddings via `SentenceTransformer("all-MiniLM-L6-v2")`.
- Hybrid embedding + numeric concatenation (embedding dims + scaled numeric features).

## Modeling

Implemented in [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb):

### TF‑IDF Pipelines

ColumnTransformer:

- `TfidfVectorizer` on `text`
- `StandardScaler` on numeric features
Classifiers evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM (probability enabled)

### Embedding Pipelines

Process:

1. Encode `text` with MiniLM sentence transformer.
2. Scale numeric features (`text_length`, `avg_word_length`).
3. Concatenate features and train classifier.

Models:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Gradient Boosting

### Evaluation

Per model:

- Accuracy
- Classification report (precision/recall/F1)
- ROC AUC
- Confusion matrix
- 5-fold cross‑validation accuracy
- ROC curve visualization

## Streamlit Web App (Interactive Demo)

A lightweight UI to classify any input summary as real or fake using the saved TF‑IDF Logistic Regression pipeline.

### Launch

Option 1: Conda (recommended)

```markdown
conda env create -f environment.yml  # creates env named 'impostor-hunt'
# If environment.yml defines a different name, adjust below:
conda activate impostor-hunt
streamlit run src/app.py
```

Then open the local URL (usually `http://localhost:8501`).  
Enter text, submit: the app computes required numeric features and feeds them to the stored Pipeline.

### App Features

- Uses stored `logistic_regression_tfidf_more_features.joblib` pipeline (no manual vectorization needed).
- Displays predicted class plus probability of each class.
- Simple, reproducible feature construction (length + avg word length).

## Test Inference & Submission

From [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb):

1. For each test pair, predict probability of "real" for both `file_1` and `file_2`.
2. Choose `real_text_id = 1` if `P(real | file_1) ≥ P(real | file_2)` else 2.
3. Save submission CSV with columns: `id`, `real_text_id`.
4. Generated submission examples in `data/submissions/`:
   - `log_reg_emb_extra_features_submission.csv`
   - `rf_emb__extra_features_submission.csv`
   - `knn_emb_submission.csv`
   - `gb_emb_submission.csv`

## Reproducibility

1. (Optional) Create environment:

   ```markdown
   python -m venv .venv
   .venv/Scripts/activate (Windows) or source .venv/bin/activate (Unix)
   pip install -U pip
   pip install pandas numpy scikit-learn sentence-transformers matplotlib
   ```

2. Run notebooks in order:
   - `notebooks/create_dataset.ipynb`
   - `notebooks/Tradition_ML.ipynb`
   - `notebooks/Modeling.ipynb`
3. Upload chosen submission file to Kaggle.

## Future Improvements

- Hyperparameter optimization (GridSearch / Optuna) for best embedding model.
- Add more linguistic / stylometric features (readability scores, punctuation ratios).
- Try pairwise / ranking approach (learning-to-rank).
- Ensemble of TF‑IDF and embedding models (probability averaging).
- Calibrate probabilities (Platt scaling / isotonic regression).

## Acknowledgments

- Kaggle competition organizers.
- `sentence-transformers` library.
