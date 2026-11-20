# Research stages vs project management stages

How aligned are the 12 research-project stages and the 12 project-management stages?

Each text file in `data/research` and `data/pm` is one stage. The **first line is the stage name**
(for example `3. Conducting a Literature Review`); every bullet after it is one item (`Label: description`).

## Run it

```powershell
pip install -r requirements.txt
python align_stages.py          # compares the items, writes results/
python app.py                   # then open http://127.0.0.1:5000
```

Always start with `python`. The first run downloads the sentence-embedding model (`all-MiniLM-L6-v2`), so it needs internet.

Other methods: `python align_stages.py --method tfidf` (no download, compares shared words only) and
`--method spacy` (needs `spacy` and `en_core_web_md`; static word vectors, weaker than MiniLM).
Use the default `minilm` for the final figures.

## What it does

1. `build_items.py` reads the text files into one table of items (`results/stage_items.csv`).
   Text damaged by find-and-replace accidents in the middle of words is repaired automatically: the letters "cre"
   (in *incremental*, *credibility*, *idea-creation*) had been replaced by a long phrase, and "safe" inside "safety"
   had been replaced by "SAFe / Scaled Agile Framework". Repaired items are flagged `repaired`.
   Items whose wording is identical on both sides (stage 12 is the same text in both lists) are flagged.
2. `align_stages.py` turns each item (label + description) into a vector and compares every research item with every
   project management item by cosine similarity (`results/item_similarity.csv`).
3. `metrics.py` computes the alignment measures. `app.py` shows them.

## Two further checks

**Whole-stage comparison.** `align_stages.py` also scores each stage as a whole: a stage's vector is the average of its items'
vectors, and two stages are compared by cosine similarity (`results/stage_matrix_whole_stage.csv`). On the Overview, choose
*Stage score based on: Whole stage* to see it. The page also shows how closely the item-based and whole-stage scores agree.
The average is used instead of one long text because sentence-embedding models cut long input off (MiniLM stops after
about 256 word pieces, and the biggest stage has 23 items).

**Language-model check.** `judge_pairs.py` asks a local model (through Ollama) whether each item and its closest match
describe the same activity, and saves *same / partly / different* with a one-sentence reason.

```powershell
ollama pull ministral-3:8b
python judge_pairs.py                         # or: --model llama3.1:8b  --top-k 2
python app.py
```

* It resumes if interrupted (verdicts are saved as it goes) and retries pairs it could not read.
* `--backend mock` writes placeholder verdicts from the similarity score so you can see the pages without a model.
  The app shows a warning while mock verdicts are present; do not use them as results.
* Needs Ollama running on this computer (`http://localhost:11434`, change with `--ollama-url`). No extra Python packages.
* Output: `results/pair_judgments.csv`. In the app: the **Language-model check** page, a verdict on the Items and Item
  pages, a summary on the Overview (how many pairs above the similarity threshold the model confirms), and a
  grid view *Language model: item pairs judged the same*.

## Measures

```
best match         = the most similar item on the other side
aligned            = best-match similarity >= the chosen threshold
coverage           = share of items that are aligned (reported for each side separately)
stage match score  = 1/2 * ( mean over research items in stage i of their best similarity inside PM stage j
                           + mean over PM items in stage j of their best similarity inside research stage i )
order match        = Spearman correlation between a research stage's number and the number of its closest PM stage
```

## Files written to `results/`

| File | Contents |
|---|---|
| `stage_items.csv` | every item with its stage, label, description, repaired and identical flags |
| `item_similarity.csv` | research items (rows) x project management items (columns) similarity |
| `run_info.json` | method, model, default threshold |
| `stage_matrix.csv` | 12 x 12 stage match scores with the full stage names on both axes (identical items left out) |
| `stage_matrix_whole_stage.csv` | the same grid scored from the average meaning of each whole stage |
| `item_embeddings.npz` | item vectors (used for the whole-stage comparison) |
| `pair_judgments.csv` | written by `judge_pairs.py`: each judged pair, the verdict and the reason |
| `item_best_matches.csv` | each item's closest item on the other side, its similarity and whether it is aligned |
| `item_stage_matches.csv` | every item against every stage of the other side: the closest item in each stage, in full, with its similarity |
| `alignment_summary.csv` | the headline figures with and without identical items |

## The web app

* **Overview**: the headline figures, the 12 x 12 stage grid (click a cell to compare two stages), and the closest stages for each stage. Change the similarity at which items count as aligned, and choose whether identical items are left out.
* **Items**: every item with its closest match, filtered by side, stage, aligned or not, or searched by word.
* **Item page** (open any item from the Items page): the item in full, then its closest item in each of the 12 stages of the other side, with full stage names, full item text and similarity.
* **Language-model check**: every judged pair with the verdict and the model's reason, filterable by verdict and stage.
* **Stage comparison**: for two stages, each item and its closest item in the other stage.
* **How it works**: the measures and cautions.

## Read with care

* The share of aligned items depends on the threshold, and a sensible value differs by method. On the Items page, sort by similarity and read the pairs around your threshold.
* Stages hold 4 to 23 items, so scores for small stages are less stable.
* Similar content is not proof that one stage is carried out by the other.
