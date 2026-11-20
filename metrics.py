"""
Alignment measures between research-stage items and project-management-stage items.
Shared by align_stages.py (writes CSV summaries) and app.py (interactive view).

Inputs: stage_items.csv and item_similarity.csv (research items x PM items, cosine similarity).

ITEM LEVEL
  best match of an item = the most similar item on the other side
  aligned               = best match similarity >= threshold
  coverage (research)   = share of research items that are aligned   (same for PM items)
  mean best similarity  = average of the best-match similarities

STAGE LEVEL (research stage i x PM stage j)
  M[i, j] = 1/2 * ( mean over research items in i of their best similarity inside stage j
                  + mean over PM items in j of their best similarity inside stage i )

ORDER
  each research stage's closest PM stage j*(i); Spearman correlation of i against j*(i), and the share of
  stages whose closest stage is the same number or one step away (the stage lists are sequential)
"""
import json
import os

import numpy as np
import pandas as pd


class Data:
    def __init__(self, items, sim, info, emb=None, judgments=None):
        self.items = items
        self.sim = sim
        self.info = info
        self.emb = emb                  # item vectors (for the whole-stage comparison)
        self.judgments = judgments      # language-model verdicts on item pairs (optional)


def load(results_dir):
    items_p = os.path.join(results_dir, "stage_items.csv")
    sim_p = os.path.join(results_dir, "item_similarity.csv")
    info_p = os.path.join(results_dir, "run_info.json")
    if not (os.path.isfile(items_p) and os.path.isfile(sim_p)):
        return None
    items = pd.read_csv(items_p, keep_default_na=False)
    for col in ("repaired", "identical_other_side"):
        items[col] = items[col].astype(str).str.lower().isin(["true", "1"])
    sim = pd.read_csv(sim_p, index_col=0)
    info = json.load(open(info_p)) if os.path.isfile(info_p) else {}
    emb = None
    emb_p = os.path.join(results_dir, "item_embeddings.npz")
    if os.path.isfile(emb_p):
        z = np.load(emb_p, allow_pickle=True)
        emb = pd.DataFrame(z["E"], index=[str(i) for i in z["ids"]])
    judg = None
    jp = os.path.join(results_dir, "pair_judgments.csv")
    if os.path.isfile(jp):
        judg = pd.read_csv(jp, keep_default_na=False)
        if judg.empty or "verdict" not in judg.columns:
            judg = None
        else:                      # a pair judged again later replaces its earlier verdict
            judg = judg.drop_duplicates(["research_id", "pm_id"], keep="last").reset_index(drop=True)
    return Data(items, sim, info, emb, judg)


def _best(S, rows, cols, other_rows):
    """Best match (index, similarity) for each row of S[rows][:, cols]."""
    sub = S[np.ix_(rows, cols)]
    j = sub.argmax(axis=1)
    return j, sub[np.arange(len(rows)), j]


def analyse(data, threshold, exclude_identical=True, basis="items"):
    items, sim = data.items, data.sim
    r_all = items[items.side == "research"].reset_index(drop=True)
    p_all = items[items.side == "pm"].reset_index(drop=True)
    r = r_all[~r_all.identical_other_side] if exclude_identical else r_all
    p = p_all[~p_all.identical_other_side] if exclude_identical else p_all
    r = r.reset_index(drop=True)
    p = p.reset_index(drop=True)
    S = sim.loc[r.item_id, p.item_id].to_numpy(dtype=float)

    # ---- item level
    jr = S.argmax(axis=1)
    best_r = r.assign(best_id=p.item_id.to_numpy()[jr], best_sim=S.max(axis=1))
    ip = S.argmax(axis=0)
    best_p = p.assign(best_id=r.item_id.to_numpy()[ip], best_sim=S.max(axis=0))
    for b in (best_r, best_p):
        b["aligned"] = b["best_sim"] >= threshold
    lookup = items.set_index("item_id")
    for b in (best_r, best_p):
        b["best_label"] = b["best_id"].map(lookup["skill_label"])
        b["best_stage_no"] = b["best_id"].map(lookup["stage_no"])
        b["best_stage"] = b["best_id"].map(lookup["stage"])

    # ---- stage level
    r_nos = sorted(r_all.stage_no.unique())
    p_nos = sorted(p_all.stage_no.unique())
    M = pd.DataFrame(np.nan, index=r_nos, columns=p_nos)
    for i in r_nos:
        ri = np.flatnonzero((r.stage_no == i).to_numpy())
        for j in p_nos:
            pj = np.flatnonzero((p.stage_no == j).to_numpy())
            if len(ri) and len(pj):
                sub = S[np.ix_(ri, pj)]
                M.at[i, j] = 0.5 * (sub.max(axis=1).mean() + sub.max(axis=0).mean())

    # whole-stage view: each stage is one vector (the average of its items' vectors)
    Mc = None
    if data.emb is not None and set(items.item_id) <= set(data.emb.index):
        Mc = pd.DataFrame(np.nan, index=r_nos, columns=p_nos)
        cr = {i: data.emb.loc[r.item_id[r.stage_no == i]].to_numpy().mean(axis=0) for i in r_nos if (r.stage_no == i).any()}
        cp = {j: data.emb.loc[p.item_id[p.stage_no == j]].to_numpy().mean(axis=0) for j in p_nos if (p.stage_no == j).any()}
        for i, a in cr.items():
            for j, b in cp.items():
                Mc.at[i, j] = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    # language-model verdicts: item pairs judged the same (1) or partly the same (0.5), counted per stage pair
    J = J_same = J_partly = None
    if data.judgments is not None:
        j_ = data.judgments
        keep_ids_r, keep_ids_p = set(r.item_id), set(p.item_id)
        j_ = j_[j_.research_id.isin(keep_ids_r) & j_.pm_id.isin(keep_ids_p)]
        stage_of = items.set_index("item_id")["stage_no"]
        J_same = pd.DataFrame(0.0, index=r_nos, columns=p_nos)
        J_partly = pd.DataFrame(0.0, index=r_nos, columns=p_nos)
        for _, row in j_.iterrows():
            tgt = J_same if row.verdict == "same" else (J_partly if row.verdict == "partly" else None)
            if tgt is not None:
                tgt.at[int(stage_of[row.research_id]), int(stage_of[row.pm_id])] += 1
        J = J_same + 0.5 * J_partly
        for i in r_nos:
            if not (r.stage_no == i).any():
                J.loc[i, :] = np.nan
        for j in p_nos:
            if not (p.stage_no == j).any():
                J.loc[:, j] = np.nan

    used = basis if ((basis == "whole" and Mc is not None) or (basis == "judged" and J is not None)) else "items"
    Msel = {"items": M, "whole": Mc, "judged": J}[used]

    def closest(frame, axis):
        out = {}
        for k in (frame.index if axis == 0 else frame.columns):
            vec = frame.loc[k] if axis == 0 else frame[k]
            vec = vec.dropna().sort_values(ascending=False)
            if used == "judged":
                vec = vec[vec > 0]
            out[k] = [(int(n), float(v)) for n, v in vec.head(3).items()]
        return out

    top_for_r = closest(Msel, 0)
    top_for_p = closest(Msel, 1)

    pairs = [(i, v[0][0]) for i, v in top_for_r.items() if v]
    rho = np.nan
    within1 = np.nan
    if len(pairs) >= 3:
        a, b = zip(*pairs)
        rho = pd.Series(a).corr(pd.Series(b), method="spearman")
        within1 = float(np.mean([abs(x - y) <= 1 for x, y in pairs]))
    pairs_p = [(j, v[0][0]) for j, v in top_for_p.items() if v]
    within1_p = float(np.mean([abs(x - y) <= 1 for x, y in pairs_p])) if pairs_p else np.nan
    diag = [float(Msel.at[i, i]) for i in r_nos if i in Msel.columns and not np.isnan(Msel.at[i, i])]
    offdiag = [float(Msel.at[i, j]) for i in r_nos for j in p_nos if i != j and not np.isnan(Msel.at[i, j])]
    agreement = None
    if Mc is not None:
        both = pd.DataFrame({"a": M.stack(), "b": Mc.stack()}).dropna()
        if len(both) > 3:
            agreement = float(both["a"].corr(both["b"]))

    summary = {
        "n_research": len(r), "n_pm": len(p),
        "n_excluded": int(len(r_all) - len(r) + len(p_all) - len(p)),
        "cov_research": float(best_r.aligned.mean()) if len(r) else np.nan,
        "cov_pm": float(best_p.aligned.mean()) if len(p) else np.nan,
        "mean_best_research": float(best_r.best_sim.mean()) if len(r) else np.nan,
        "mean_best_pm": float(best_p.best_sim.mean()) if len(p) else np.nan,
        "n_aligned_research": int(best_r.aligned.sum()), "n_aligned_pm": int(best_p.aligned.sum()),
        "spearman": None if pd.isna(rho) else float(rho),
        "within1_research": None if pd.isna(within1) else within1,
        "within1_pm": None if pd.isna(within1_p) else within1_p,
        "mean_diagonal": float(np.mean(diag)) if diag else np.nan,
        "mean_offdiagonal": float(np.mean(offdiag)) if offdiag else np.nan,
        "agreement_items_vs_whole": agreement,
    }
    stage_names = {
        "research": r_all.drop_duplicates("stage_no").set_index("stage_no")["stage"].to_dict(),
        "pm": p_all.drop_duplicates("stage_no").set_index("stage_no")["stage"].to_dict(),
    }
    return {"S": S, "r": r, "p": p, "best_r": best_r, "best_p": best_p, "M": M, "Mc": Mc, "J": J,
            "J_same": J_same, "J_partly": J_partly, "Msel": Msel, "basis": used,
            "top_for_r": top_for_r, "top_for_p": top_for_p, "summary": summary, "stage_names": stage_names}


def item_vs_stages(data, item_id):
    """One item against every stage of the other side: the closest item inside each stage, in full."""
    items = data.items.set_index("item_id", drop=False)
    x = items.loc[item_id]
    row = data.sim.loc[item_id] if x.side == "research" else data.sim[item_id]
    other = items.loc[row.index]
    rows = []
    for stage_no, grp in other.groupby("stage_no"):
        sims = row[grp.index]
        best = sims.idxmax()
        y = items.loc[best]
        rows.append({
            "stage_no": int(stage_no), "stage": grp["stage"].iloc[0], "n_items": len(grp),
            "best_id": best, "best_label": y.skill_label, "best_desc": y.description,
            "sim": float(sims[best]), "ident": bool(y.identical_other_side),
        })
    return x, pd.DataFrame(rows)


def item_stage_table(data):
    """Every item against every stage of the other side (for Excel)."""
    out = []
    for iid in data.items["item_id"]:
        x, t = item_vs_stages(data, iid)
        for _, y in t.iterrows():
            out.append({"item_id": iid, "side": x.side, "stage_no": x.stage_no, "stage": x.stage,
                        "skill_label": x.skill_label, "other_stage_no": y.stage_no, "other_stage": y.stage,
                        "closest_item_id": y.best_id, "closest_label": y.best_label,
                        "closest_description": y.best_desc, "similarity": y.sim})
    return pd.DataFrame(out)


def judgment_lookup(data):
    """{(research_id, pm_id): (verdict, reason)} for pairs the language model has judged."""
    j = data.judgments
    if j is None:
        return {}
    return {(r.research_id, r.pm_id): (r.verdict, r.reason) for r in j.itertuples()}


def judged_summary(data, res, threshold):
    """How the language model's verdicts compare with the similarity scores."""
    j = data.judgments
    if j is None:
        return None
    ids_r, ids_p = set(res["r"].item_id), set(res["p"].item_id)
    j = j[j.research_id.isin(ids_r) & j.pm_id.isin(ids_p) & j.verdict.isin(["same", "partly", "different"])]
    if j.empty:
        return None
    look = {(a, b): v for a, b, v in zip(j.research_id, j.pm_id, j.verdict)}
    out = {"n_pairs": len(j), "models": sorted(set(data.judgments["judge_model"].astype(str))),
           "counts": j.verdict.value_counts().to_dict()}
    above = j[j.similarity.astype(float) >= threshold]
    out["n_above"] = len(above)
    out["above_confirmed"] = float(above.verdict.isin(["same", "partly"]).mean()) if len(above) else None
    out["above_same"] = float((above.verdict == "same").mean()) if len(above) else None
    for side, best in (("research", res["best_r"]), ("pm", res["best_p"])):
        n = len(best)
        c = {"same": 0, "partly": 0, "different": 0}
        for iid, bid in zip(best.item_id, best.best_id):
            key = (iid, bid) if side == "research" else (bid, iid)
            v = look.get(key)
            if v in c:
                c[v] += 1
        out[side] = {"n": n, **c, "not_judged": n - sum(c.values())}
    return out
