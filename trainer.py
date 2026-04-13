import re

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

MODELS = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=42, dual="auto"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}


def _score_matrix(model, X):
    """Return (n_samples, n_classes) probability/decision array, or None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if s.ndim == 2:
            return s
    return None


# ── Classification ─────────────────────────────────────────────────────────────

def train_and_evaluate(df, text_col, label_col, selected_models, test_size=0.2,
                       progress_callback=None):
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values

    min_class_count = np.bincount(np.unique(y, return_inverse=True)[1]).min()
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results, pipelines, y_preds = {}, {}, {}

    for i, name in enumerate(selected_models):
        if progress_callback:
            progress_callback(i, len(selected_models), name)

        model = clone(MODELS[name])
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        y_preds[name] = y_pred

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }
        pipelines[name] = (model, vectorizer)

    return results, pipelines, y_test, y_preds


def predict_all(pipelines, text):
    return {name: model.predict(vec.transform([text]))[0]
            for name, (model, vec) in pipelines.items()}


# ── Next-word prediction ───────────────────────────────────────────────────────

def _tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def build_ngram_pairs(texts, window_size=3):
    """Slide a window over each passage to produce (context, next_word) pairs."""
    X, y = [], []
    for text in texts:
        words = _tokenize(text)
        for i in range(window_size, len(words)):
            X.append(" ".join(words[i - window_size: i]))
            y.append(words[i])
    return X, y


def train_predictor(df, text_col, selected_models, window_size=3,
                    test_size=0.2, progress_callback=None):
    texts = df[text_col].astype(str).tolist()
    X, y = build_ngram_pairs(texts, window_size)

    if len(X) < 50:
        raise ValueError(f"Only {len(X)} n-gram pairs generated — add more text to the corpus.")

    X_arr, y_arr = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=42
    )

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1),
                                  sublinear_tf=True, min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    k = min(5, len(np.unique(y_train)))
    results, pipelines, y_preds = {}, {}, {}

    for i, name in enumerate(selected_models):
        if progress_callback:
            progress_callback(i, len(selected_models), name)

        model = clone(MODELS[name])
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        y_preds[name] = y_pred

        top1 = round(accuracy_score(y_test, y_pred), 4)
        topk_val = None
        try:
            scores = _score_matrix(model, X_test_vec)
            if scores is not None:
                # Restrict to test words the model was trained on
                known = np.isin(y_test, model.classes_)
                if known.sum() >= k:
                    topk_val = round(
                        top_k_accuracy_score(
                            y_test[known], scores[known],
                            k=k, labels=model.classes_,
                        ), 4
                    )
        except Exception:
            pass

        results[name] = {
            "Top-1 Accuracy": top1,
            f"Top-{k} Accuracy": topk_val if topk_val is not None else "—",
        }
        pipelines[name] = (model, vectorizer)

    corpus_stats = {
        "total_pairs": len(X),
        "train_pairs": len(X_train),
        "test_pairs": len(X_test),
        "vocab_size": len(vectorizer.vocabulary_),
        "window_size": window_size,
        "k": k,
    }
    return results, pipelines, y_test, y_preds, corpus_stats


def predict_next_topk(pipelines, seed_text, window_size=3, top_k=5):
    """Return top-k next-word candidates per model for the given seed."""
    words = _tokenize(seed_text)
    if not words:
        return {}
    context = " ".join(words[-window_size:])
    result = {}
    for name, (model, vectorizer) in pipelines.items():
        vec = vectorizer.transform([context])
        scores = _score_matrix(model, vec)
        if scores is not None:
            k = min(top_k, scores.shape[1])
            top_idx = scores[0].argsort()[-k:][::-1]
            result[name] = [model.classes_[i] for i in top_idx]
        else:
            result[name] = [model.predict(vec)[0]]
    return result


def generate_text(pipelines, seed_text, window_size=3, steps=10):
    """Each model independently generates `steps` words beyond the seed."""
    base = _tokenize(seed_text)
    if not base:
        return {}
    outputs = {}
    for name, (model, vectorizer) in pipelines.items():
        words = list(base)
        for _ in range(steps):
            context = " ".join(words[-window_size:])
            try:
                vec = vectorizer.transform([context])
                words.append(model.predict(vec)[0])
            except Exception:
                break
        outputs[name] = words
    return outputs
