# ML Text Lab

A lightweight, interactive machine learning workbench for experimenting with English text.
Train five scikit-learn models side-by-side, compare their performance metrics, and explore
two distinct NLP tasks — all from a Streamlit UI with no coding required.

---

## Features

### Classification
Label text into predefined categories. Compare models across Accuracy, Precision, Recall, and F1 Score.

| Built-in dataset | Classes | Rows |
|---|---|---|
| **Text Topics** | sports, technology, health, politics | 195 |
| **Exception Types** | TypeError, ValueError, KeyError, IndexError, AttributeError, FileNotFoundError, ImportError, ZeroDivisionError | 80 |

Upload any CSV with a text column and a label column to use your own data.

### Next Word Prediction
Train models as n-gram language models. Given a seed phrase, each model predicts the most
likely next word(s). Includes a 200-passage built-in corpus (~3,400 words) covering daily life,
nature, science, history, food, travel, business, arts, and education.

- **Generate Text** — each model independently extends a seed phrase word-by-word
- **Next Word Race** — see all models' top-K candidates for a given phrase side by side

---

## Models

All five models are trained and compared simultaneously. Each uses TF-IDF vectorization.

| Model | Notes |
|---|---|
| **Naive Bayes** | Fast baseline; strong on text frequency features |
| **Logistic Regression** | Reliable linear classifier; good calibrated probabilities |
| **Linear SVM** | Often best for text classification with TF-IDF |
| **Random Forest** | Ensemble; handles non-linear patterns |
| **Gradient Boosting** | Boosted ensemble; powerful but slower to train |

---

## Getting Started

### Prerequisites

- Python 3.12+
- pip

### Install and run

```bash
git clone https://github.com/<your-username>/ml-text-lab.git
cd ml-text-lab

pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run with Docker

```bash
docker build -t ml-text-lab .
docker run -p 8501:8501 ml-text-lab
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
ml-text-lab/
├── app.py                        # Streamlit UI — all pages and interactivity
├── trainer.py                    # ML logic — training, evaluation, prediction
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
└── data/
    ├── sample_data.csv           # Text topic classification (195 rows, 4 classes)
    ├── exceptions_data.csv       # Exception type classification (80 rows, 8 classes)
    └── prediction_corpus.csv     # Next word prediction corpus (200 passages)
```

---

## Usage Guide

### Classification mode

1. Select **Classification** in the sidebar Task radio.
2. Pick a built-in dataset or upload a CSV.
3. Choose the text and label columns (auto-selected for built-ins).
4. Adjust the test split percentage (default 20%).
5. Check/uncheck models to include in the run.
6. Click **Train & Compare Models**.

**Results include:**
- Metrics table — Accuracy, Precision (weighted), Recall (weighted), F1 Score (weighted); best per column is highlighted
- Bar chart — compare any metric across models
- Confusion matrix — per-model heatmap showing where misclassifications occur
- Try a Prediction — enter any text to see what all trained models predict

### Next Word Prediction mode

1. Select **Next Word Prediction** in the sidebar Task radio.
2. Pick the built-in corpus or upload a CSV with a text column.
3. Set the **context window** (2, 3, or 4 words used as input to predict the next word).
4. Click **Train Prediction Models**.

**Results include:**
- Stats row — total n-gram pairs, training pairs, vocabulary size, test pairs
- Metrics table — Top-1 Accuracy and Top-K Accuracy per model
- Bar chart — Top-1 accuracy comparison
- **Generate Text** tab — enter a seed phrase, choose how many words to generate; each model outputs its own continuation
- **Next Word Race** tab — enter a phrase and see each model's top-K candidate next words ranked side by side

> **Note on accuracy:** Next-word prediction is inherently hard. With a vocabulary of ~1,300 words,
> a Top-1 accuracy of 7–8% and Top-5 accuracy of 25–30% is meaningful — random chance would be ~0.1%.
> Models improve significantly with more training text.

---

## Using Your Own Data

### Classification

Prepare a CSV with at least two columns:

```
text,label
"The team won the championship last night.",sports
"A new vaccine entered phase three trials.",health
```

Upload via **Upload CSV** in the sidebar. Any column names work — you select them in the UI.

### Next Word Prediction

Prepare a CSV with a single text column of English passages, one passage per row:

```
text
"The morning sun rose slowly over the quiet hills."
"She packed her bag and headed to the station early."
```

More text = larger vocabulary = better predictions. Aim for at least 100 passages.

---

## Architecture

```
app.py  ──────────────────────────────────────────────────────────────────────
  │  Streamlit UI
  │  ├── Sidebar: mode selector, dataset picker, model checkboxes
  │  ├── Classification section  ──►  train_and_evaluate()
  │  └── Prediction section      ──►  train_predictor()
  │                                   generate_text()
  │                                   predict_next_topk()
  ▼
trainer.py  ──────────────────────────────────────────────────────────────────
  │
  ├── Classification
  │   ├── TF-IDF (1–2 grams, max 10,000 features, sublinear TF)
  │   ├── Stratified train/test split
  │   └── Weighted Precision / Recall / F1
  │
  └── Next Word Prediction
      ├── Sliding window → (context string, next word) pairs
      ├── TF-IDF (unigram, sublinear TF)
      ├── Top-1 and Top-K accuracy (evaluated on known-vocabulary test items)
      └── Autoregressive generation (each model chains its own predictions)
```

---

## CI Pipeline

Every push and pull request to `main`/`master` runs the following checks via GitHub Actions:

| Step | What it checks |
|---|---|
| **Syntax check** | `py_compile` on `app.py` and `trainer.py` |
| **Import check** | All public trainer functions importable |
| **Smoke test — topics** | Classification on `sample_data.csv`; all models produce F1 > 0 |
| **Smoke test — exceptions** | Classification on `exceptions_data.csv`; all models produce F1 > 0 |
| **Smoke test — prediction** | Corpus generates >500 pairs, generation and candidate functions return results |
| **Docker build** | Image builds successfully end-to-end |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web UI |
| scikit-learn | ≥ 1.4 | All ML models and metrics |
| pandas | ≥ 2.0 | Data loading and table display |
| numpy | ≥ 1.26 | Array operations |
| matplotlib | ≥ 3.8 | Bar charts |
| seaborn | ≥ 0.13 | Confusion matrix heatmap |

---

## License

MIT
