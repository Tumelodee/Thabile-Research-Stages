"""
Turn the stage text files into one table of items.

Each text file describes one stage. The FIRST LINE is the stage name (for example
"3. Conducting a Literature Review"); every bullet line after it is one item
("Label: description").

    python build_items.py --research-dir data/research --pm-dir data/pm --out results/stage_items.csv
"""
import argparse
import os
import re

import pandas as pd

BULLETS = ("\u2022", "-", "*", "\u00b7", "\u2013")
# Some project management files contain find-and-replace accidents in the middle of words. They are put back:
#   "cre" (incremental, credibility, idea-creation) became the first phrase below,
#   "safe" inside "safety" became "SAFe / Scaled Agile Framework" (so "safety" read "SAFe / Scaled Agile Frameworkty").
CORRUPT = "CRE / Commercial Real Estate project-based management"
FIXES = [
    ("CRE / COMMERCIAL REAL ESTATE PROJECT-BASED MANAGEMENT", "CRE"),
    (CORRUPT, "cre"),
    ("SAFe / Scaled Agile Frameworkty", "safety"),
    ("SAFE / SCALED AGILE FRAMEWORKTY", "SAFETY"),
]


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def parse_file(path, side, fallback_no):
    with open(path, encoding="utf-8-sig") as fh:
        lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    if not lines:
        return []
    m = re.match(r"^(\d+)\s*[.)]\s*(.+)$", lines[0])
    stage_no = int(m.group(1)) if m else fallback_no
    stage = m.group(2).strip() if m else lines[0]

    bodies = []
    for line in lines[1:]:
        if line[0] in BULLETS:
            bodies.append(line[1:].strip())
        elif bodies:                      # a wrapped line belongs to the previous bullet
            bodies[-1] += " " + line
    rows = []
    for k, body in enumerate(bodies, start=1):
        repaired = any(bad in body for bad, _ in FIXES)
        for bad, good in FIXES:
            body = body.replace(bad, good)
        label, sep, desc = body.partition(":")
        if not sep or len(label) > 140:
            label, desc = body, ""
        label, desc = label.strip(), desc.strip()
        if repaired and label:
            label = label[0].upper() + label[1:]
        rows.append({
            "item_id": f"{'R' if side == 'research' else 'P'}{stage_no:02d}.{k:02d}",
            "side": side, "stage_no": stage_no, "stage": stage, "item_no": k,
            "skill_label": label, "description": desc,
            "text": f"{label}: {desc}" if desc else label,
            "repaired": repaired, "source_file": os.path.basename(path),
        })
    return rows


def build(research_dir, pm_dir):
    rows = []
    for side, folder in (("research", research_dir), ("pm", pm_dir)):
        files = sorted((f for f in os.listdir(folder) if f.lower().endswith(".txt")), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"No .txt files found in {folder}")
        for k, name in enumerate(files, start=1):
            rows += parse_file(os.path.join(folder, name), side, k)
    df = pd.DataFrame(rows)

    # identical wording on both sides (a copied stage) is flagged so it can be left out of the headline figures
    other = {s: set(_norm(t) for t in df.loc[df.side == s, "text"]) for s in ("research", "pm")}
    df["identical_other_side"] = [
        _norm(t) in other["pm" if s == "research" else "research"] for t, s in zip(df["text"], df["side"])
    ]
    for side in ("research", "pm"):
        nums = sorted(df.loc[df.side == side, "stage_no"].unique())
        if nums != list(range(1, len(nums) + 1)):
            print(f"WARNING: {side} stage numbers are {nums} (expected 1..{len(nums)}).")
    return df


def main():
    ap = argparse.ArgumentParser(description="Parse stage text files into stage_items.csv")
    ap.add_argument("--research-dir", default="data/research")
    ap.add_argument("--pm-dir", default="data/pm")
    ap.add_argument("--out", default="results/stage_items.csv")
    args = ap.parse_args()
    df = build(args.research_dir, args.pm_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    for side in ("research", "pm"):
        d = df[df.side == side]
        print(f"{side}: {d.stage_no.nunique()} stages, {len(d)} items")
    print(f"Repaired text in {int(df.repaired.sum())} items; "
          f"{int(df.identical_other_side.sum())} items have identical wording on the other side.")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
