import json
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag import retrieve

def recall_at_k(gold_pages, retrieved_pages):
    gold = set(gold_pages)
    return 1 if any(p in gold for p in retrieved_pages) else 0

def main():
    path = Path("/Users/aryanjha/Desktop/paper-rag-assistant/eval/questions.jsonl")
    assert path.exists(), "Missing eval/questions.jsonl"

    ks = [1, 3, 5, 8]
    totals = {k: 0 for k in ks}
    n = 0

    per_paper = {}  # paper -> {k: hits, n}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            source_file = ex["source_file"]
            q = ex["question"]
            gold_pages = ex["gold_pages"]

            n += 1
            per_paper.setdefault(source_file, {"n": 0, **{k: 0 for k in ks}})
            per_paper[source_file]["n"] += 1

            for k in ks:
                ctxs = retrieve(q, top_k=k, source_file=source_file)
                retrieved_pages = [c["page"] for c in ctxs]
                hit = recall_at_k(gold_pages, retrieved_pages)

                totals[k] += hit
                per_paper[source_file][k] += hit

    print(f"\nEvaluated {n} questions\n")
    for k in ks:
        print(f"Recall@{k}: {totals[k] / n:.3f} ({totals[k]}/{n})")

    print("\nPer paper:")
    for paper, stats in per_paper.items():
        print(f"\n- {paper} (n={stats['n']})")
        for k in ks:
            print(f"  Recall@{k}: {stats[k] / stats['n']:.3f} ({stats[k]}/{stats['n']})")

if __name__ == "__main__":
    main()