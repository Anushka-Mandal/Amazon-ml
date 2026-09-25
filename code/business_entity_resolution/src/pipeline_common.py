"""Shared steps: context features, chunked pair-feature building, match decision."""
import gc
import os
import time

import numpy as np
import polars as pl

from config import work
from features import context_features, id_num, load_records, pair_features, token_idf


def build_features(split, s1_filter=None, tag="all", chunk_s1=60000):
    """Compute features for candidate pairs of the given split; writes work/{split}_{tag}_f*.parquet.

    s1_filter: optional Series/list of S1 ids to restrict to (context features still use the
    whole candidate set so competition statistics are complete).
    """
    t0 = time.time()
    cand = pl.read_parquet(work(f"{split}_blocks.parquet"))
    cand = cand.with_columns(id_num("s1_id").alias("s1n"), id_num("r_id", "src").alias("rn")
                             ).drop("s1_id", "r_id")
    s1, r = load_records(split)
    cand = context_features(cand, s1)
    # small per-record id tables used to map integer keys back to entity ids chunk by chunk
    s1map = s1.select(id_num("id").alias("s1n"), pl.col("id").alias("s1_id"))
    rmap = r.select(id_num("id", pl.col("id").str.slice(1, 1)).alias("rn"),
                    pl.col("id").alias("r_id"))
    if s1_filter is not None:
        keep = pl.DataFrame({"s1_id": s1_filter}).select(id_num("s1_id").alias("s1n"))
        cand = cand.join(keep, on="s1n", how="semi")
    gc.collect()
    idf = token_idf(s1, r)
    ids = cand["s1n"].unique().sort()
    files = []
    for ci, lo in enumerate(range(0, len(ids), chunk_s1)):
        fn = work(f"{split}_{tag}_f{ci:03d}.parquet")
        files.append(fn)
        if os.path.exists(fn):
            continue
        part = (cand.join(pl.DataFrame({"s1n": ids[lo:lo + chunk_s1]}), on="s1n", how="semi")
                .join(s1map, on="s1n", how="left").join(rmap, on="rn", how="left")
                .drop("s1n", "rn"))
        part = part.select(["s1_id", "r_id"] + [c for c in part.columns if c not in ("s1_id", "r_id")])
        feats = pair_features(part, s1, r, idf)
        feats.write_parquet(fn)
        del feats, part
        gc.collect()
        print(f"  features {split}/{tag} chunk {ci} ({lo + chunk_s1}/{len(ids)}) "
              f"{time.time() - t0:.0f}s", flush=True)
    return files


def decide(scored, t_abs, t_rel=0.0, exclusive=True):
    """scored: (s1_id, r_id, p). Returns chosen (s1_id, r_id) links.

    exclusive: a Source 2/3 record is linked to at most one Source 1 entity (its best).
    t_abs: minimum probability; t_rel: minimum ratio to the best probability of that S1.
    """
    d = scored
    if exclusive:
        d = d.filter(pl.col("p") >= pl.col("p").max().over("r_id"))
        d = d.unique(subset=["r_id"], keep="first")
    d = d.filter(pl.col("p") >= t_abs)
    if t_rel > 0:
        d = d.filter(pl.col("p") >= t_rel * pl.col("p").max().over("s1_id"))
    return d.select("s1_id", "r_id")


def write_outputs(s1_ids, cand, links, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = pl.DataFrame({"source1_entity_id": s1_ids})

    def agg(df, col):
        g = (df.unique(subset=["s1_id", "r_id"]).sort("s1_id", "r_id")
             .group_by("s1_id", maintain_order=True)
             .agg(pl.col("r_id").str.join(",").alias(col))
             .rename({"s1_id": "source1_entity_id"}))
        return base.join(g, on="source1_entity_id", how="left").with_columns(
            pl.col(col).fill_null(""))

    m = agg(links, "matched_entity_ids")
    c = agg(cand.select("s1_id", "r_id"), "candidate_entity_ids")
    for df, name in ((m, "matching_results.tsv"), (c, "candidate_pairs.tsv")):
        with open(os.path.join(out_dir, name), "w") as f:
            f.write("\t".join(df.columns) + "\n")
            for a, b in df.iter_rows():
                f.write(f"{a}\t{b}\n")
    return m, c
