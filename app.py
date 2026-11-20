"""
Web view of the research-stage vs project-management-stage alignment.

    python align_stages.py        # once, writes results/
    python app.py                 # then open http://127.0.0.1:5000
"""
import argparse
import os

import numpy as np
import pandas as pd
from flask import Flask, abort, render_template, request, url_for

import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))

app = Flask(__name__)

LIGHT = (238, 243, 246)
DARK = (8, 58, 74)
_CACHE = {}


def get_data():
    paths = [os.path.join(RESULTS_DIR, n) for n in ("stage_items.csv", "item_similarity.csv", "run_info.json")]
    key = tuple(os.path.getmtime(p) if os.path.isfile(p) else 0 for p in paths)
    hit = _CACHE.get("data")
    if hit and hit[0] == key:
        return hit[1]
    data = metrics.load(RESULTS_DIR)
    _CACHE["data"] = (key, data)
    return data


def heat(t):
    t = float(np.clip(t, 0, 1))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(LIGHT, DARK))
    lum = sum(w * (c / 255) ** 2.2 for w, c in zip((0.2126, 0.7152, 0.0722), rgb))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", ("#ffffff" if lum < 0.2 else "#06101c")


def settings(data):
    default = float(data.info.get("default_threshold", 0.5))
    try:
        thr = float(request.args.get("threshold", default))
    except ValueError:
        thr = default
    thr = min(0.99, max(0.05, thr))
    excl = request.args.get("identical", "exclude") != "include"
    return thr, excl


def keep(thr, excl, **extra):
    out = {"threshold": f"{thr:g}", "identical": "exclude" if excl else "include", **extra}
    basis = request.args.get("basis", "items")
    if basis != "items" and "basis" not in out:
        out["basis"] = basis
    return out


def base_ctx(data, thr=None, excl=None):
    info = data.info if data else {}
    kp = keep(thr, excl) if thr is not None else {}
    has_j = bool(data is not None and data.judgments is not None)
    mock = bool(has_j and (data.judgments["judge_model"].astype(str) == "mock").any())
    return {"info": info, "standin": bool(info) and info.get("method") != "minilm", "thr": thr, "excl": excl, "kp": kp,
            "has_judg": has_j, "judge_mock": mock}


@app.template_filter("p0")
def p0(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.0f}"


@app.route("/")
def overview():
    data = get_data()
    if data is None:
        return render_template("empty.html", results_dir=RESULTS_DIR, **base_ctx(None))
    thr, excl = settings(data)
    req_basis = request.args.get("basis", "items")
    res = analyse_cached(data, thr, excl, req_basis)
    basis = res["basis"]
    Msel = res["Msel"]
    names = res["stage_names"]
    finite = Msel.to_numpy(dtype=float)
    finite = finite[~np.isnan(finite)]
    if basis == "judged":
        lo, hi = 0.0, float(finite.max()) if len(finite) and finite.max() > 0 else 1.0
    else:
        lo, hi = (np.percentile(finite, 5), np.percentile(finite, 98)) if len(finite) else (0, 1)
    span = (hi - lo) or 1.0
    rows = []
    for i in Msel.index:
        cells = []
        best_j = res["top_for_r"][i][0][0] if res["top_for_r"][i] else None
        for j in Msel.columns:
            v = Msel.at[i, j]
            if np.isnan(v):
                cells.append({"nan": True, "title": "Left out: identical wording in both lists"})
                continue
            bg, fg = heat((v - lo) / span)
            col_best = res["top_for_p"][j][0][0] if res["top_for_p"][j] else None
            if basis == "judged":
                s_, p_ = int(res["J_same"].at[i, j]), int(res["J_partly"].at[i, j])
                text = str(s_ + p_) if (s_ + p_) else ""
                title = f"{s_} pair(s) judged the same, {p_} partly the same"
            else:
                text = f"{100 * v:.0f}"
                title = f"Research stage {i} and project management stage {j}: {100 * v:.0f}"
            cells.append({"nan": False, "text": text, "bg": bg, "fg": fg, "best": best_j == j and bool(text),
                          "colbest": col_best == i and bool(text),
                          "url": url_for("pair", r=i, p=j, **keep(thr, excl)), "title": title})
        rows.append({"no": i, "name": names["research"][i],
                     "url": url_for("items", side="research", stage=i, **keep(thr, excl)), "cells": cells})
    cols = [{"no": j, "name": names["pm"][j], "url": url_for("items", side="pm", stage=j, **keep(thr, excl))}
            for j in Msel.columns]

    def top_list(top, other):
        out = []
        for k, lst in top.items():
            out.append({"no": k, "hits": [{"no": n, "name": other[n], "v": v} for n, v in lst]})
        return out

    bases = [("items", "Items: best-matching items")]
    if res["Mc"] is not None:
        bases.append(("whole", "Whole stage: average meaning of all its items"))
    if data.judgments is not None:
        bases.append(("judged", "Language model: item pairs judged the same"))
    s_items = analyse_cached(data, thr, excl, "items")["summary"]
    return render_template(
        "overview.html", s=s_items, s_sel=res["summary"], basis=basis, bases=bases, rows=rows, cols=cols, names=names,
        top_r=top_list(res["top_for_r"], names["pm"]), top_p=top_list(res["top_for_p"], names["research"]),
        form={"threshold": f"{thr:g}", "identical": "exclude" if excl else "include", "basis": basis},
        jsum=metrics.judged_summary(data, analyse_cached(data, thr, excl, "items"), thr),
        n_ident=int(data.items.identical_other_side.sum()), **base_ctx(data, thr, excl))


def analyse_cached(data, thr, excl, basis="items"):
    key = ("res", round(thr, 4), excl, basis, id(data))
    if key not in _CACHE:
        _CACHE[key] = metrics.analyse(data, thr, excl, basis)
    return _CACHE[key]


@app.route("/items")
def items():
    data = get_data()
    if data is None:
        return render_template("empty.html", results_dir=RESULTS_DIR, **base_ctx(None))
    thr, excl = settings(data)
    res = analyse_cached(data, thr, excl)
    side = request.args.get("side", "research")
    side = side if side in ("research", "pm") else "research"
    stage = request.args.get("stage", "all")
    status = request.args.get("status", "all")
    sort = request.args.get("sort", "stage")
    q = request.args.get("q", "").strip()
    df = (res["best_r"] if side == "research" else res["best_p"]).copy()
    if stage != "all" and stage.isdigit():
        df = df[df.stage_no == int(stage)]
    if status == "aligned":
        df = df[df.aligned]
    elif status == "unaligned":
        df = df[~df.aligned]
    if q:
        df = df[df.skill_label.str.contains(q, case=False, regex=False) | df.description.str.contains(q, case=False, regex=False)]
    if sort == "high":
        df = df.sort_values("best_sim", ascending=False)
    elif sort == "low":
        df = df.sort_values("best_sim", ascending=True)
    else:
        df = df.sort_values(["stage_no", "item_no"])
    look = metrics.judgment_lookup(data)
    rows = []
    for _, x in df.iterrows():
        key = (x.item_id, x.best_id) if side == "research" else (x.best_id, x.item_id)
        verdict = look.get(key, (None, ""))
        r_no = int(x.stage_no) if side == "research" else int(x.best_stage_no)
        p_no = int(x.best_stage_no) if side == "research" else int(x.stage_no)
        rows.append({
            "id": x.item_id, "stage_no": int(x.stage_no), "stage": x.stage, "label": x.skill_label,
            "desc": x.description, "best_id": x.best_id, "best_label": x.best_label,
            "best_stage_no": int(x.best_stage_no), "best_stage": x.best_stage, "sim": float(x.best_sim),
            "aligned": bool(x.aligned), "ident": bool(x.identical_other_side), "repaired": bool(x.repaired),
            "verdict": verdict[0], "reason": verdict[1],
            "pair_url": url_for("pair", r=r_no, p=p_no, **keep(thr, excl)),
            "item_url": url_for("item", id=x.item_id, **keep(thr, excl)),
        })
    stage_names = res["stage_names"][side]
    form = {"side": side, "stage": stage, "status": status, "sort": sort, "q": q,
            "threshold": f"{thr:g}", "identical": "exclude" if excl else "include"}
    other = "pm" if side == "research" else "research"
    return render_template("items.html", rows=rows, form=form, side=side, other=other,
                           stage_names=stage_names, total_side=len(res["best_r"] if side == "research" else res["best_p"]),
                           n_hidden=int(data.items[(data.items.side == side) & data.items.identical_other_side].shape[0]) if excl else 0,
                           **base_ctx(data, thr, excl))


@app.route("/item")
def item():
    data = get_data()
    if data is None:
        return render_template("empty.html", results_dir=RESULTS_DIR, **base_ctx(None))
    thr, excl = settings(data)
    iid = request.args.get("id", "")
    if iid not in set(data.items.item_id):
        abort(404)
    x, t = metrics.item_vs_stages(data, iid)
    best_i = int(t["sim"].idxmax())
    look = metrics.judgment_lookup(data)
    rows = []
    for k, y in t.iterrows():
        key = (iid, y.best_id) if x.side == "research" else (y.best_id, iid)
        vd = look.get(key, (None, ""))
        r_no = int(x.stage_no) if x.side == "research" else int(y.stage_no)
        p_no = int(y.stage_no) if x.side == "research" else int(x.stage_no)
        rows.append({**y.to_dict(), "is_best": k == best_i, "aligned": bool(y.sim >= thr),
                     "verdict": vd[0], "reason": vd[1],
                     "pair_url": url_for("pair", r=r_no, p=p_no, **keep(thr, excl)),
                     "item_url": url_for("item", id=y.best_id, **keep(thr, excl))})
    order = t.sort_values("sim", ascending=False)
    second = order.iloc[1] if len(order) > 1 else None
    other = "pm" if x.side == "research" else "research"
    return render_template("item.html", x=x.to_dict(), rows=rows, best=rows[best_i], second=second,
                           other=other, back=url_for("items", side=x.side, stage=int(x.stage_no), **keep(thr, excl)),
                           **base_ctx(data, thr, excl))


@app.route("/pair")
def pair():
    data = get_data()
    if data is None:
        return render_template("empty.html", results_dir=RESULTS_DIR, **base_ctx(None))
    thr, excl = settings(data)
    res = analyse_cached(data, thr, excl)
    try:
        r_no, p_no = int(request.args.get("r")), int(request.args.get("p"))
    except (TypeError, ValueError):
        abort(404)
    names = res["stage_names"]
    if r_no not in names["research"] or p_no not in names["pm"]:
        abort(404)
    r, p, S = res["r"], res["p"], res["S"]
    ri = np.flatnonzero((r.stage_no == r_no).to_numpy())
    pj = np.flatnonzero((p.stage_no == p_no).to_numpy())
    left, right = [], []
    if len(ri) and len(pj):
        sub = S[np.ix_(ri, pj)]
        for a, k in enumerate(sub.argmax(axis=1)):
            x, y = r.iloc[ri[a]], p.iloc[pj[k]]
            left.append({"id": x.item_id, "label": x.skill_label, "desc": x.description,
                         "m_id": y.item_id, "m_label": y.skill_label, "m_desc": y.description, "sim": float(sub[a, k])})
        for b, k in enumerate(sub.argmax(axis=0)):
            x, y = p.iloc[pj[b]], r.iloc[ri[k]]
            right.append({"id": x.item_id, "label": x.skill_label, "desc": x.description,
                          "m_id": y.item_id, "m_label": y.skill_label, "m_desc": y.description, "sim": float(sub[k, b])})
    left.sort(key=lambda d: -d["sim"])
    right.sort(key=lambda d: -d["sim"])
    score = res["M"].at[r_no, p_no]
    return render_template("pair.html", r_no=r_no, p_no=p_no, r_name=names["research"][r_no], p_name=names["pm"][p_no],
                           left=left, right=right, score=None if np.isnan(score) else float(score),
                           back=url_for("overview", **keep(thr, excl)), **base_ctx(data, thr, excl))


@app.route("/judgments")
def judgments():
    data = get_data()
    if data is None:
        return render_template("empty.html", results_dir=RESULTS_DIR, **base_ctx(None))
    thr, excl = settings(data)
    ctx = base_ctx(data, thr, excl)
    if data.judgments is None:
        return render_template("judgments.html", ready=False, **ctx)
    items = data.items.set_index("item_id")
    j = data.judgments.copy()
    if excl:
        ident = items["identical_other_side"]
        j = j[~(j.research_id.map(ident).fillna(False) | j.pm_id.map(ident).fillna(False))]
    verdict = request.args.get("verdict", "all")
    stage = request.args.get("stage", "all")
    sort = request.args.get("sort", "high")
    if verdict in ("same", "partly", "different", "unparsed", "error"):
        j = j[j.verdict == verdict]
    if stage.isdigit():
        j = j[j.research_id.map(items["stage_no"]) == int(stage)]
    j = j.assign(similarity=j.similarity.astype(float))
    j = j.sort_values("similarity", ascending=(sort == "low"))
    rows = []
    for _, x in j.iterrows():
        a, b = items.loc[x.research_id], items.loc[x.pm_id]
        rows.append({"r": a.to_dict(), "p": b.to_dict(), "sim": float(x.similarity), "verdict": x.verdict,
                     "reason": x.reason, "model": x.judge_model,
                     "pair_url": url_for("pair", r=int(a.stage_no), p=int(b.stage_no), **keep(thr, excl))})
    all_j = data.judgments
    counts = all_j.verdict.value_counts().to_dict()
    stage_names = data.items[data.items.side == "research"].drop_duplicates("stage_no").set_index("stage_no")["stage"].to_dict()
    form = {"verdict": verdict, "stage": stage, "sort": sort, "threshold": f"{thr:g}",
            "identical": "exclude" if excl else "include"}
    return render_template("judgments.html", ready=True, rows=rows, counts=counts, form=form,
                           stage_names=stage_names, models=sorted(set(all_j.judge_model.astype(str))), **ctx)


@app.route("/method")
def method():
    data = get_data()
    thr, excl = settings(data) if data else (None, None)
    return render_template("method.html", n_ident=int(data.items.identical_other_side.sum()) if data else 0,
                           n_rep=int(data.items.repaired.sum()) if data else 0, **base_ctx(data, thr, excl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="View the stage alignment results in the browser.")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    RESULTS_DIR = os.path.abspath(a.results_dir)
    print(f"Results folder: {RESULTS_DIR}")
    print(f"Open http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, debug=False)
