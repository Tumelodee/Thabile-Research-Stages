"""
Ask a language model whether each item and its closest match describe the same activity.

Needs Ollama running locally (https://ollama.com) with a model pulled, for example:
    ollama pull ministral-3:8b
    python judge_pairs.py                                  # judges each item's closest match
    python judge_pairs.py --model llama3.1:8b --top-k 2    # another model, two closest matches per item
    python judge_pairs.py --backend mock                   # NO language model: placeholder verdicts to test the app only

For every candidate pair the model answers  same / partly / different  with a one-sentence reason.
Results are appended to results/pair_judgments.csv as it goes, so an interrupted run resumes where it stopped.
Run align_stages.py first.
"""
import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

import pandas as pd

import metrics

PROMPT_VERSION = "pair-judge-v1"

SYSTEM = (
    "You compare activities taken from two different lists: the stages of a research project and the stages of "
    "project management. You decide whether two items describe the same activity or skill. Judge the meaning, "
    "not the wording. Answer only with a JSON object."
)

USER = """Item A comes from a list of RESEARCH PROJECT stages.
Stage: {a_stage}
Item: {a_label}
Description: {a_desc}

Item B comes from a list of PROJECT MANAGEMENT stages.
Stage: {b_stage}
Item: {b_label}
Description: {b_desc}

Do A and B describe the same activity or skill?
- "same": essentially the same activity, skill or task, even if worded differently.
- "partly": they overlap (one is broader or narrower, or they share only one component).
- "different": different activities, even if some of the words are alike.

Answer with JSON only: {{"verdict": "same" or "partly" or "different", "reason": "one short sentence"}}"""

VERDICTS = {
    "same": "same", "match": "same", "identical": "same", "equivalent": "same",
    "partly": "partly", "partial": "partly", "partially": "partly", "overlap": "partly", "related": "partly",
    "different": "different", "no": "different", "unrelated": "different", "not the same": "different",
}


def parse_answer(text):
    """Return (verdict, reason) from the model's reply; verdict is 'unparsed' when it cannot be read."""
    obj = None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*?\}", str(text), flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except ValueError:
                obj = None
    if not isinstance(obj, dict):
        return "unparsed", str(text)[:300]
    raw = str(obj.get("verdict", "")).strip().lower()
    return VERDICTS.get(raw, "unparsed"), str(obj.get("reason", "")).strip()[:400]


def ollama_chat(url, model, user, timeout):
    body = json.dumps({
        "model": model, "stream": False, "format": "json", "options": {"temperature": 0},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"]


def candidate_pairs(data, top_k, min_sim, include_identical):
    items = data.items.set_index("item_id")
    S = data.sim
    found = {}
    for rid in S.index:
        for pid in S.loc[rid].nlargest(top_k).index:
            found[(rid, pid)] = float(S.at[rid, pid])
    for pid in S.columns:
        for rid in S[pid].nlargest(top_k).index:
            found[(rid, pid)] = float(S.at[rid, pid])
    rows = []
    for (rid, pid), sim in found.items():
        if sim < min_sim:
            continue
        if not include_identical and (items.at[rid, "identical_other_side"] or items.at[pid, "identical_other_side"]):
            continue
        rows.append((rid, pid, sim))
    return sorted(rows, key=lambda t: -t[2])


def main():
    ap = argparse.ArgumentParser(description="Language-model check of the closest item pairs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--model", default="ministral-3:8b", help="Ollama model name (default: ministral-3:8b)")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--top-k", type=int, default=1, help="Closest matches per item to judge (default 1)")
    ap.add_argument("--min-sim", type=float, default=None,
                    help="Skip pairs below this similarity (default: 0.15 under the app's starting threshold)")
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="Judge only the first N pairs (for a quick test)")
    ap.add_argument("--include-identical", action="store_true", help="Also judge items identical in both lists")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    data = metrics.load(args.results_dir)
    if data is None:
        sys.exit("No results found. Run  python align_stages.py  first.")
    default_thr = float(data.info.get("default_threshold", 0.5))
    min_sim = args.min_sim if args.min_sim is not None else max(0.0, default_thr - 0.15)
    pairs = candidate_pairs(data, args.top_k, min_sim, args.include_identical)[: args.max_pairs]
    if args.limit:
        pairs = pairs[: args.limit]
    model_name = "mock" if args.backend == "mock" else args.model

    out_path = os.path.join(args.results_dir, "pair_judgments.csv")
    done = set()
    if os.path.isfile(out_path):
        old = pd.read_csv(out_path, keep_default_na=False)
        done = {(a, b) for a, b, v, m in zip(old.research_id, old.pm_id, old.verdict, old.judge_model)
                if m == model_name and v in ("same", "partly", "different")}
    todo = [p for p in pairs if (p[0], p[1]) not in done]
    print(f"{len(pairs)} candidate pairs (similarity >= {min_sim:.2f}); {len(done)} already judged; {len(todo)} to do.")
    if args.backend == "mock":
        print("MOCK backend: verdicts come from the similarity score, NOT from a language model.")

    items = data.items.set_index("item_id")
    new_file = not os.path.isfile(out_path)
    counts = {}
    with open(out_path, "a", encoding="utf-8", newline="") as fh:
        import csv
        cols = ["pair_id", "research_id", "pm_id", "similarity", "verdict", "reason", "judge_model",
                "prompt_version", "judged_at"]
        w = csv.DictWriter(fh, fieldnames=cols)
        if new_file:
            w.writeheader()
        for k, (rid, pid, sim) in enumerate(todo, start=1):
            a, b = items.loc[rid], items.loc[pid]
            if args.backend == "mock":
                verdict = "same" if sim >= default_thr + 0.1 else ("partly" if sim >= default_thr - 0.1 else "different")
                reason = "MOCK verdict from the similarity score; no language model was run."
            else:
                prompt = USER.format(a_stage=a.stage, a_label=a.skill_label, a_desc=a.description or "(none)",
                                     b_stage=b.stage, b_label=b.skill_label, b_desc=b.description or "(none)")
                verdict, reason = "error", ""
                for attempt in range(3):
                    try:
                        verdict, reason = parse_answer(ollama_chat(args.ollama_url, args.model, prompt, args.timeout))
                        break
                    except urllib.error.HTTPError as e:
                        msg = e.read().decode("utf-8", "ignore")[:200]
                        if e.code == 404:
                            sys.exit(f"Model '{args.model}' was not found by Ollama. Run:  ollama pull {args.model}")
                        verdict, reason = "error", f"HTTP {e.code}: {msg}"
                    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                        verdict, reason = "error", str(e)[:200]
                        if k == 1 and attempt == 0 and isinstance(e, urllib.error.URLError) and not done:
                            sys.exit(f"Could not reach Ollama at {args.ollama_url}. Start it (ollama serve) and try again.\n{e}")
            w.writerow({"pair_id": f"{rid}|{pid}", "research_id": rid, "pm_id": pid, "similarity": round(sim, 4),
                        "verdict": verdict, "reason": reason, "judge_model": model_name,
                        "prompt_version": PROMPT_VERSION,
                        "judged_at": datetime.datetime.now().isoformat(timespec="seconds")})
            fh.flush()
            counts[verdict] = counts.get(verdict, 0) + 1
            if k % 10 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} judged  {counts}")
    print(f"\nDone. Verdicts this run: {counts}")
    print("Saved to", out_path, "- reload the app to see them.")
    bad = counts.get("unparsed", 0) + counts.get("error", 0)
    if bad:
        print(f"{bad} pairs could not be judged (unparsed or error); run the same command again to retry them.")


if __name__ == "__main__":
    main()
