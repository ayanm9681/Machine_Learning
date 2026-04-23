import io
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from trainer import (MODELS, generate_text, predict_all, predict_next_topk,
                     train_and_evaluate, train_predictor)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LexiScope", layout="wide")
st.title("LexiScope")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CLF_DATASETS = {
    "Text Topics — 4 categories, 195 rows": {
        "path": os.path.join(DATA_DIR, "sample_data.csv"),
        "text_col": "text", "label_col": "category",
    },
    "Exception Types — 8 types, 80 rows": {
        "path": os.path.join(DATA_DIR, "exceptions_data.csv"),
        "text_col": "description", "label_col": "exception_type",
    },
}
PRED_CORPUS_PATH = os.path.join(DATA_DIR, "prediction_corpus.csv")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Top-level mode selector
    mode = st.radio(
        "Task",
        ["Classification", "Next Word Prediction"],
        help="Classification — label text into categories.  "
             "Next Word Prediction — generate the most likely next words.",
    )
    st.divider()

    # ── Classification sidebar ─────────────────────────────────────────────────
    if mode == "Classification":
        source = st.radio("Dataset", list(CLF_DATASETS.keys()) + ["Upload CSV"])

        clf_df = None
        default_text, default_label = None, None

        if source == "Upload CSV":
            up = st.file_uploader("CSV file", type="csv")
            if up:
                clf_df = pd.read_csv(up)
                st.success(f"{len(clf_df)} rows loaded.")
        else:
            meta = CLF_DATASETS[source]
            clf_df = pd.read_csv(meta["path"])
            default_text, default_label = meta["text_col"], meta["label_col"]

        clf_text_col = clf_label_col = None
        clf_selected: list[str] = []
        clf_test_size = 0.2

        if clf_df is not None:
            cols = clf_df.columns.tolist()
            t_idx = cols.index(default_text) if default_text in cols else 0
            clf_text_col = st.selectbox("Text column", cols, index=t_idx)
            rem = [c for c in cols if c != clf_text_col]
            l_idx = rem.index(default_label) if default_label in rem else 0
            clf_label_col = st.selectbox("Label column", rem, index=l_idx)
            clf_test_size = st.slider("Test split %", 10, 40, 20) / 100

            st.subheader("Models")
            clf_selected = [n for n, v in
                            {n: st.checkbox(n, value=True) for n in MODELS}.items() if v]

        st.divider()
        st.subheader("Save / Load")
        clf_pkl = st.file_uploader("Load saved model (.pkl)", type="pkl", key="clf_pkl_upload")
        if clf_pkl is not None:
            try:
                saved = pickle.load(clf_pkl)
                st.session_state.update(
                    clf_results=saved["results"], clf_pipes=saved["pipes"],
                    clf_y_test=saved["y_test"], clf_y_preds=saved["y_preds"],
                )
                st.success("Model loaded — scroll down to see results.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    # ── Prediction sidebar ─────────────────────────────────────────────────────
    else:
        pred_source = st.radio("Corpus", ["Built-in corpus (200 passages)", "Upload CSV"])

        pred_df = None
        pred_text_col = "text"

        if pred_source == "Upload CSV":
            up = st.file_uploader("CSV file (needs a text column)", type="csv")
            if up:
                pred_df = pd.read_csv(up)
                st.success(f"{len(pred_df)} rows loaded.")
                pred_text_col = st.selectbox("Text column", pred_df.columns.tolist())
        else:
            pred_df = pd.read_csv(PRED_CORPUS_PATH)

        pred_selected: list[str] = []
        pred_window = 3
        pred_test_size = 0.2

        if pred_df is not None:
            pred_window = st.select_slider(
                "Context window (words)",
                options=[2, 3, 4],
                value=3,
                help="How many preceding words to use as context for the prediction.",
            )
            pred_test_size = st.slider("Test split %", 10, 30, 20, key="pred_split") / 100

            st.subheader("Models")
            pred_selected = [n for n, v in
                             {n: st.checkbox(n, value=True, key=f"p_{n}") for n in MODELS}.items()
                             if v]

        st.divider()
        st.subheader("Save / Load")
        pred_pkl = st.file_uploader("Load saved model (.pkl)", type="pkl", key="pred_pkl_upload")
        if pred_pkl is not None:
            try:
                saved = pickle.load(pred_pkl)
                st.session_state.update(
                    pred_results=saved["results"], pred_pipes=saved["pipes"],
                    pred_y_test=saved["y_test"], pred_y_preds=saved["y_preds"],
                    pred_stats=saved["stats"],
                )
                st.success("Model loaded — scroll down to see results.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Classification":
    st.caption("Train multiple classifiers on labeled text and compare Accuracy, Precision, Recall, and F1.")

    if clf_df is None:
        st.info("Choose a dataset in the sidebar to get started.")
        st.stop()

    # Dataset overview
    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", len(clf_df))
    c2.metric("Classes", clf_df[clf_label_col].nunique())
    c3.metric("Test samples", max(1, int(len(clf_df) * clf_test_size)))
    c4.metric("Models selected", len(clf_selected))

    with st.expander("Preview & class distribution"):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.dataframe(clf_df[[clf_text_col, clf_label_col]].head(10), use_container_width=True)
        with col_b:
            fig, ax = plt.subplots(figsize=(4, 3))
            counts = clf_df[clf_label_col].value_counts()
            ax.barh(counts.index, counts.values, color="#555555", edgecolor="white")
            ax.set_xlabel("Count")
            ax.set_title("Class Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Train
    if st.button("Train & Compare Models", type="primary", use_container_width=True):
        if not clf_selected:
            st.warning("Select at least one model in the sidebar.")
        else:
            bar = st.progress(0, text="Starting…")
            try:
                results, pipes, y_test, y_preds = train_and_evaluate(
                    clf_df, clf_text_col, clf_label_col, clf_selected, clf_test_size,
                    progress_callback=lambda i, n, name: bar.progress(
                        int(i / n * 100), text=f"Training {name}…"),
                )
                bar.progress(100, text="Done!")
                st.session_state.update(
                    clf_results=results, clf_pipes=pipes,
                    clf_y_test=y_test, clf_y_preds=y_preds,
                )
                buf = io.BytesIO()
                pickle.dump({"results": results, "pipes": pipes,
                             "y_test": y_test, "y_preds": y_preds}, buf)
                st.download_button(
                    "Download trained models (.pkl)", data=buf.getvalue(),
                    file_name="clf_models.pkl", mime="application/octet-stream",
                    key="clf_dl",
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                bar.empty()

    # Results
    if "clf_results" not in st.session_state:
        st.stop()

    results = st.session_state["clf_results"]
    pipes   = st.session_state["clf_pipes"]
    y_test  = st.session_state["clf_y_test"]
    y_preds = st.session_state["clf_y_preds"]

    st.divider()
    st.subheader("Results")

    res_df = (pd.DataFrame(results).T.reset_index()
              .rename(columns={"index": "Model"}))
    best = res_df.loc[res_df["F1 Score"].idxmax(), "Model"]
    st.success(f"Best model by F1: **{best}** — {results[best]['F1 Score']:.4f}")

    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]

    def _bold_max(s):
        return ["font-weight: 700" if v == s.max() else "" for v in s]

    st.dataframe(
        res_df.style
        .apply(_bold_max, subset=metric_cols)
        .format({c: "{:.4f}" for c in metric_cols}),
        use_container_width=True, hide_index=True,
    )

    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Confusion Matrix", "Try a Prediction"])

    with tab1:
        metric = st.selectbox("Metric", metric_cols, index=3)
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#333333" if m == best else "#888888" for m in res_df["Model"]]
        bars = ax.bar(res_df["Model"], res_df[metric], color=colors, edgecolor="white", width=0.55)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — Model Comparison  (dark = best)")
        avg = res_df[metric].mean()
        ax.axhline(avg, color="gray", linestyle="--", linewidth=1, label=f"Average: {avg:.3f}")
        for bar, val in zip(bars, res_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        cm_model = st.selectbox("Model", list(results.keys()), key="cm_sel")
        classes = sorted(set(y_test))
        cm = confusion_matrix(y_test, y_preds[cm_model], labels=classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
                    xticklabels=classes, yticklabels=classes,
                    ax=ax, linewidths=0.4, linecolor="white")
        ax.set_xlabel("Predicted", labelpad=8)
        ax.set_ylabel("Actual", labelpad=8)
        ax.set_title(f"Confusion Matrix — {cm_model}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.write("Enter any text and see what each trained model predicts.")
        user_text = st.text_area("Text", height=100,
                                  placeholder="e.g. The researchers published a study on neural networks…")
        if st.button("Predict", key="clf_predict"):
            if not user_text.strip():
                st.warning("Enter some text first.")
            else:
                preds = predict_all(pipes, user_text)
                pred_df = pd.DataFrame(preds.items(), columns=["Model", "Predicted Class"])
                mode_class = pred_df["Predicted Class"].mode()[0]

                def _hi(row):
                    weight = "font-weight: 700" if row["Predicted Class"] == mode_class else ""
                    return ["", weight]

                st.dataframe(pred_df.style.apply(_hi, axis=1),
                             use_container_width=True, hide_index=True)
                if pred_df["Predicted Class"].nunique() == 1:
                    st.success(f"All models agree: **{mode_class}**")
                else:
                    st.info(f"Models disagree. Most common: **{mode_class}**")


# ══════════════════════════════════════════════════════════════════════════════
# NEXT WORD PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.caption("Train classifiers as n-gram language models to predict the most likely next word in a sentence.")

    if pred_df is None:
        st.info("Choose a corpus in the sidebar to get started.")
        st.stop()

    # Corpus overview
    st.subheader("Corpus")
    c1, c2, c3 = st.columns(3)
    c1.metric("Passages", len(pred_df))
    c2.metric("Context window", f"{pred_window} words")
    c3.metric("Models selected", len(pred_selected))

    with st.expander("Preview corpus"):
        st.dataframe(pred_df[[pred_text_col]].head(10), use_container_width=True)

    # Train
    if st.button("Train Prediction Models", type="primary", use_container_width=True):
        if not pred_selected:
            st.warning("Select at least one model in the sidebar.")
        else:
            bar = st.progress(0, text="Building n-gram pairs…")
            try:
                p_results, p_pipes, p_y_test, p_y_preds, p_stats = train_predictor(
                    pred_df, pred_text_col, pred_selected, pred_window, pred_test_size,
                    progress_callback=lambda i, n, name: bar.progress(
                        int(i / n * 100), text=f"Training {name}…"),
                )
                bar.progress(100, text="Done!")
                st.session_state.update(
                    pred_results=p_results, pred_pipes=p_pipes,
                    pred_y_test=p_y_test, pred_y_preds=p_y_preds,
                    pred_stats=p_stats,
                )
                buf = io.BytesIO()
                pickle.dump({"results": p_results, "pipes": p_pipes,
                             "y_test": p_y_test, "y_preds": p_y_preds,
                             "stats": p_stats}, buf)
                st.download_button(
                    "Download trained models (.pkl)", data=buf.getvalue(),
                    file_name="pred_models.pkl", mime="application/octet-stream",
                    key="pred_dl",
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                bar.empty()

    if "pred_results" not in st.session_state:
        st.stop()

    p_results = st.session_state["pred_results"]
    p_pipes   = st.session_state["pred_pipes"]
    p_y_test  = st.session_state["pred_y_test"]
    p_y_preds = st.session_state["pred_y_preds"]
    p_stats   = st.session_state["pred_stats"]
    k         = p_stats["k"]

    # Stats row
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("N-gram pairs", f"{p_stats['total_pairs']:,}")
    s2.metric("Training pairs", f"{p_stats['train_pairs']:,}")
    s3.metric("Vocabulary size", f"{p_stats['vocab_size']:,}")
    s4.metric("Test pairs", f"{p_stats['test_pairs']:,}")

    st.subheader("Results")
    p_res_df = (pd.DataFrame(p_results).T.reset_index()
                .rename(columns={"index": "Model"}))

    top1_col = "Top-1 Accuracy"
    topk_col = f"Top-{k} Accuracy"
    best_pred = p_res_df.loc[p_res_df[top1_col].idxmax(), "Model"]

    st.success(f"Best model by Top-1 Accuracy: **{best_pred}** — {p_results[best_pred][top1_col]:.4f}")
    st.caption(f"Note: next-word prediction is hard — even a top-1 accuracy of 15–30% is meaningful "
               f"given a vocabulary of {p_stats['vocab_size']:,} words.")

    num_cols = [c for c in p_res_df.columns if c != "Model" and p_res_df[c].apply(
        lambda x: isinstance(x, (int, float))).all()]

    def _bold_max_pred(s):
        return ["font-weight: 700" if v == s.max() else "" for v in s]

    st.dataframe(
        p_res_df.style
        .apply(_bold_max_pred, subset=num_cols)
        .format({c: "{:.4f}" for c in num_cols}),
        use_container_width=True, hide_index=True,
    )

    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Generate Text", "Next Word Race"])

    # ── Tab 1: bar chart ───────────────────────────────────────────────────────
    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#333333" if m == best_pred else "#888888" for m in p_res_df["Model"]]
        bars = ax.bar(p_res_df["Model"], p_res_df[top1_col],
                      color=colors, edgecolor="white", width=0.55)
        ax.set_ylim(0, max(p_res_df[top1_col]) * 1.3)
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_title("Top-1 Accuracy — Model Comparison  (dark = best)")
        avg = p_res_df[top1_col].mean()
        ax.axhline(avg, color="gray", linestyle="--", linewidth=1, label=f"Average: {avg:.3f}")
        for bar, val in zip(bars, p_res_df[top1_col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + avg * 0.04,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: generate text ───────────────────────────────────────────────────
    with tab2:
        st.write("Each model independently continues the seed phrase one word at a time.")
        seed = st.text_input(
            "Seed phrase",
            value="the morning sun rises over",
            help=f"Use at least {pred_window} words so every model has a full context window.",
        )
        steps = st.slider("Words to generate", 5, 25, 12, key="gen_steps")

        if st.button("Generate", key="gen_btn"):
            if not seed.strip():
                st.warning("Enter a seed phrase.")
            else:
                outputs = generate_text(p_pipes, seed, p_stats["window_size"], steps)
                seed_len = len(seed.split())  # rough word count for display split

                for name, word_list in outputs.items():
                    original = " ".join(word_list[:seed_len])
                    generated = " ".join(word_list[seed_len:])
                    with st.container(border=True):
                        st.markdown(f"**{name}**")
                        st.markdown(f"{original} **{generated}**")

    # ── Tab 3: next word race ──────────────────────────────────────────────────
    with tab3:
        st.write(f"Shows each model's top-{k} candidate next words for your phrase.")
        phrase = st.text_input(
            "Phrase",
            value="the children laughed as they",
            help="The last few words are used as context.",
            key="race_phrase",
        )

        if st.button("Get Candidates", key="race_btn"):
            if not phrase.strip():
                st.warning("Enter a phrase.")
            else:
                candidates = predict_next_topk(p_pipes, phrase, p_stats["window_size"], top_k=k)
                rows = []
                for model_name, words in candidates.items():
                    row = {"Model": model_name}
                    for rank, word in enumerate(words, 1):
                        row[f"#{rank}"] = word
                    rows.append(row)
                cand_df = pd.DataFrame(rows)
                st.dataframe(cand_df, use_container_width=True, hide_index=True)

                # Show which word most models agree on
                all_top1 = [w[0] for w in candidates.values() if w]
                if all_top1:
                    from collections import Counter
                    top_agreed = Counter(all_top1).most_common(1)[0][0]
                    st.info(f"Most models predict next word: **{top_agreed}**")
