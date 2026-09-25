"""Learn native-script -> Latin dictionaries from the TRAINING ground truth only.

1. Name tokens: an Indic-script name in Source 2/3 is a word-by-word transliteration of the
   Source 1 name, so aligning tokens positionally (when token counts agree) gives a token
   dictionary (e.g. "प्राइवेट" -> "private").
2. Address segments: fully non-Latin comma-separated address segments (usually a state or city
   written in native script) are mapped to the Source 1 address segment they co-occur with most.

No external data is used; everything comes from train_ground_truth.tsv and the train sources.
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl

from config import work
from io_utils import read_ground_truth, read_source

NONLATIN = r"[ऀ-෿]"


def main():
    s1 = read_source("train", 1).select(
        pl.col("entity_id").alias("s1_id"),
        pl.col("business_name").alias("n1"),
        pl.col("business_address").alias("a1"),
    )
    gt = read_ground_truth()
    others = []
    for k in (2, 3):
        s = read_source("train", k).filter(
            pl.col("business_name").str.contains(NONLATIN)
            | pl.col("business_address").str.contains(NONLATIN)
        )
        others.append(s)
    oth = pl.concat(others).rename({"entity_id": "r_id"})
    j = oth.join(gt, on="r_id").join(s1, on="s1_id")
    print("pairs with non-latin text:", j.height)

    tok = defaultdict(Counter)
    seg = defaultdict(Counter)
    for n2, a2, n1, a1 in zip(j["business_name"].to_list(), j["business_address"].to_list(),
                              j["n1"].to_list(), j["a1"].to_list()):
        if re.search(NONLATIN, n2):
            ta = n2.split()
            tb = re.sub(r"[^\w\s]", " ", n1.lower()).split()
            if len(ta) == len(tb):
                for x, y in zip(ta, tb):
                    if re.search(NONLATIN, x):
                        tok[x][y] += 1
        if re.search(NONLATIN, a2):
            s1segs = {s.strip().lower() for s in a1.split(",") if s.strip()}
            for s in a2.split(","):
                s = s.strip()
                if s and re.search(NONLATIN, s) and not re.search(r"[A-Za-z0-9]", s):
                    for y in s1segs:
                        seg[s][y] += 1
                    seg[s]["__n__"] += 1

    tok_map = {}
    for x, c in tok.items():
        y, n = c.most_common(1)[0]
        if n >= 1 and n / sum(c.values()) >= 0.4:
            tok_map[x] = y
    seg_map = {}
    for x, c in seg.items():
        tot = c.pop("__n__")
        if tot < 3:
            continue
        # prefer the most frequent co-occurring segment that is not a pure number
        cands = [(y, n) for y, n in c.most_common(5) if not re.fullmatch(r"[\d\W]+", y)]
        if cands and cands[0][1] / tot >= 0.5:
            seg_map[x] = cands[0][0]
    print("name tokens:", len(tok_map), "address segments:", len(seg_map))
    with open(work("translit.json"), "w") as f:
        json.dump({"tokens": tok_map, "segments": seg_map}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
