"""
Measure how well research-project stages and project-management stages line up.

    python align_stages.py                       # sentence embeddings (all-MiniLM-L6-v2), default
    python align_stages.py --method tfidf        # no model download; compares shared words only
    python align_stages.py --method spacy        # needs spacy + en_core_web_md (static word vectors)

Writes to results/:
    stage_items.csv, item_similarity.csv, run_info.json         (used by the web app)
    stage_matrix.csv, stage_matrix_whole_stage.csv, item_best_matches.csv, item_stage_matches.csv,
    alignment_summary.csv   (ready to open in Excel)
    item_embeddings.npz   (used for the whole-stage comparison)
"""
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

import build_items
import metrics

# Where "related" starts differs by method, so each gets its own starting threshold (adjustable in the app).
DEFAULT_THRESHOLD = {"minilm": 0.50, "spacy": 0.60, "tfidf": 0.25}


def embed(texts, method, model_name):
    if method == "minilm":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(list(texts)).toarray()
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if method == "spacy":
        import spacy
        nlp = spacy.load(model_name)
        rows = []
        for doc in nlp.pipe(list(texts)):
            vecs = [t.vector for t in doc if t.has_vector and not (t.is_stop or t.is_punct or t.is_space)]
            rows.append(np.mean(vecs, axis=0) if vecs else np.zeros(nlp.vocab.vectors_length))
        X = np.vstack(rows)
        X = X - X.mean(axis=0, keepdims=True)      # remove the component every text shares, so scores spread out
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    raise ValueError(f"Unknown method {method}")


def main():
    ap = argparse.ArgumentParser(description="Alignment between research stages and project management stages")
    ap.add_argument("--research-dir", default="data/research")
    ap.add_argument("--pm-dir", default="data/pm")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--method", choices=["minilm", "tfidf", "spacy"], default="minilm")
    ap.add_argument("--model", default=None,
                    help="Model name (default: all-MiniLM-L6-v2 for minilm, en_core_web_md for spacy)")
    args = ap.parse_args()
    model_name = args.model or {"minilm": "all-MiniLM-L6-v2", "spacy": "en_core_web_md", "tfidf": "tf-idf (words and word pairs)"}[args.method]

    items = build_items.build(args.research_dir, args.pm_dir)
    print(f"{(items.side == 'research').sum()} research items, {(items.side == 'pm').sum()} PM items")
    print(f"Embedding with {args.method} ({model_name})...")
    E = embed(items["text"], args.method, model_name)
    emb = pd.DataFrame(E, index=items["item_id"])
    r_ids = items.loc[items.side == "research", "item_id"]
    p_ids = items.loc[items.side == "pm", "item_id"]
    sim = pd.DataFrame(emb.loc[r_ids].to_numpy() @ emb.loc[p_ids].to_numpy().T, index=r_ids, columns=p_ids)

    os.makedirs(args.results_dir, exist_ok=True)
    out = lambda name: os.path.join(args.results_dir, name)
    items.to_csv(out("stage_items.csv"), index=False)
    np.savez_compressed(out("item_embeddings.npz"), ids=items["item_id"].to_numpy().astype(str), E=E.astype(np.float32))
    sim.round(4).to_csv(out("item_similarity.csv"))
    info = {"method": args.method, "model": model_name, "default_threshold": DEFAULT_THRESHOLD[args.method],
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(info, open(out("run_info.json"), "w"), indent=2)

    data = metrics.load(args.results_dir)
    thr = DEFAULT_THRESHOLD[args.method]
    rows = []
    for excl in (True, False):
        res = metrics.analyse(data, thr, exclude_identical=excl)
        s = res["summary"]
        tag = "excluding identical items" if excl else "including identical items"
        for k, v in s.items():
            rows.append({"setting": tag, "threshold": thr, "metric": k, "value": v})
        if excl:
            names = items.drop_duplicates(["side", "stage_no"]).set_index(["side", "stage_no"])["stage"]
            M = res["M"].round(4).copy()
            M.index = [f"{i}. {names[('research', i)]}" for i in M.index]
            M.columns = [f"{j}. {names[('pm', j)]}" for j in M.columns]
            M.to_csv(out("stage_matrix.csv"), index_label="Research stage (rows) / project management stage (columns)")
            if res["Mc"] is not None:
                Mc = res["Mc"].round(4).copy()
                Mc.index, Mc.columns = M.index, M.columns
                Mc.to_csv(out("stage_matrix_whole_stage.csv"), index_label="Research stage (rows) / project management stage (columns)")
    pd.DataFrame(rows).to_csv(out("alignment_summary.csv"), index=False)

    metrics.item_stage_table(data).round({"similarity": 4}).to_csv(out("item_stage_matches.csv"), index=False)

    res = metrics.analyse(data, thr, exclude_identical=False)
    best = pd.concat([
        res["best_r"].assign(side="research"), res["best_p"].assign(side="pm")
    ])[["item_id", "side", "stage_no", "stage", "skill_label", "identical_other_side",
        "best_id", "best_label", "best_stage_no", "best_stage", "best_sim", "aligned"]]
    best.round({"best_sim": 4}).to_csv(out("item_best_matches.csv"), index=False)

    s = metrics.analyse(data, thr, exclude_identical=True)["summary"]
    print(f"\nAt similarity >= {thr} (identical items excluded):")
    print(f"  {100 * s['cov_research']:.0f}% of research items have a counterpart among the PM items")
    print(f"  {100 * s['cov_pm']:.0f}% of PM items have a counterpart among the research items")
    print(f"  mean best-match similarity: {s['mean_best_research']:.2f} (research to PM), {s['mean_best_pm']:.2f} (PM to research)")
    if s.get("agreement_items_vs_whole") is not None:
        print(f"  item-based and whole-stage scores agree with correlation {s['agreement_items_vs_whole']:.2f}")
    print(f"\nSaved results to {args.results_dir}/")
    print("Optional next step: python judge_pairs.py   (a language model checks the closest pairs)")
    print("Then run: python app.py")


if __name__ == "__main__":
    main()
