"""Candidate generation (blocking).

Every record is turned into a bag of "blocking items":
    n:  phonetic skeleton of each core-name token          (typo / transliteration tolerant)
    nb: adjacent skeleton-token bigrams of the core name
    c:  concatenated core name                            (catches web-style names)
    d:  house / plot numbers in the address
    a:  address word tokens
    ab: adjacent address-token bigrams                    (e.g. "8825 winterbrook")
Items are weighted by IDF computed on the Source 2+3 side and items that are too frequent
(df > CAP) are dropped.  Within each country label, the Source 1 x Source k similarity is a
sparse matrix product (sum of IDF of shared items) and the top-K records per source are kept
for each Source 1 record.  Countries are processed as an open set of labels.
"""
import math
import sys
import time
from multiprocessing import get_context

import numpy as np
import polars as pl
import scipy.sparse as sp

from config import N_JOBS, work

CAP = 3000          # max document frequency (Source 2+3, per country) of a usable item
TOPK = 12           # candidates kept per Source 1 record per source
MIN_SCORE = 0.0

_G = {}


def _tokens(df, col, prefix):
    return (
        df.select("row", pl.col(col).str.split(" ").alias("t"))
        .explode("t")
        .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() > 0))
        .with_columns(pl.int_range(pl.len()).over("row").alias("pos"))
        .with_columns(pl.lit(prefix).alias("p"))
    )


def _bigrams(tf, prefix):
    nxt = tf.select("row", (pl.col("pos") - 1).alias("pos"), pl.col("t").alias("t2"))
    return tf.join(nxt, on=["row", "pos"]).select(
        "row", (pl.col("t") + "_" + pl.col("t2")).alias("t"), pl.lit(prefix).alias("p"))


def make_items(df):
    """df has columns row, skel, concat, nums, alpha, addr -> (row, h) unique item hashes."""
    parts = []
    n = _tokens(df, "skel", "n")
    parts.append(n.select("row", "t", "p"))
    parts.append(_bigrams(n, "nb"))
    parts.append(df.select("row", pl.col("concat").alias("t"), pl.lit("c").alias("p"))
                 .filter(pl.col("t").str.len_chars() > 3))
    parts.append(_tokens(df, "nums", "d").select("row", "t", "p"))
    parts.append(_tokens(df, "alpha", "a").filter(pl.col("t").str.len_chars() >= 3)
                 .select("row", "t", "p"))
    parts.append(_bigrams(_tokens(df, "addr", "x"), "ab"))
    it = pl.concat(parts)
    return (it.select("row", (pl.col("p") + ":" + pl.col("t")).hash().alias("h"))
            .unique())


def _topk_chunk(args):
    lo, hi = args
    X1 = _G["X1"][lo:hi]
    res = []
    for src, XT in _G["XT"].items():
        P = (X1 @ XT).tocsr()
        if P.nnz == 0:
            continue
        counts = np.diff(P.indptr)
        rows = np.repeat(np.arange(hi - lo, dtype=np.int64), counts)
        order = np.lexsort((-P.data, rows))
        data = P.data[order]
        cols = P.indices[order]
        rank = np.arange(len(order)) - P.indptr[rows]
        keep = (rank < TOPK) & (data > MIN_SCORE)
        res.append((src, rows[keep] + lo, cols[keep], data[keep].astype(np.float32),
                    rank[keep].astype(np.int16)))
    return res


def block_split(split):
    t0 = time.time()
    s1 = pl.read_parquet(work(f"{split}_s1.parquet"))
    rs = {k: pl.read_parquet(work(f"{split}_s{k}.parquet")) for k in (2, 3)}
    cols = ["id", "country", "skel", "concat", "nums", "alpha", "addr"]
    out = []
    countries = s1["country"].unique().to_list()
    for c in countries:
        a = s1.filter(pl.col("country") == c).select(cols).with_row_index("row")
        rk = {k: r.filter(pl.col("country") == c).select(cols).with_row_index("row")
              for k, r in rs.items()}
        if a.height == 0 or all(v.height == 0 for v in rk.values()):
            continue
        ia = make_items(a)
        ir = {k: make_items(v) for k, v in rk.items()}
        dfr = pl.concat([v.select("h") for v in ir.values()]).group_by("h").len("df")
        n_r = sum(v.height for v in rk.values())
        keep = (dfr.filter(pl.col("df") <= CAP)
                .join(ia.select("h").unique(), on="h", how="semi")
                .with_row_index("j")
                .with_columns((np.log((n_r + 1) / (pl.col("df") + 1))).alias("idf")))
        V = keep.height
        a_it = ia.join(keep, on="h")
        X1 = sp.csr_matrix(
            (a_it["idf"].to_numpy().astype(np.float32),
             (a_it["row"].to_numpy(), a_it["j"].to_numpy())), shape=(a.height, V))
        XT = {}
        for k, v in ir.items():
            r_it = v.join(keep.select("h", "j"), on="h")
            XT[k] = sp.csr_matrix(
                (np.ones(r_it.height, dtype=np.float32),
                 (r_it["j"].to_numpy(), r_it["row"].to_numpy())), shape=(V, rk[k].height))
        selfscore = np.asarray(X1.sum(axis=1)).ravel()
        del ia, ir, a_it
        _G["X1"], _G["XT"] = X1, XT
        print(f"[{split}:{c}] s1={a.height} vocab={V} nnz1={X1.nnz} "
              f"build {time.time()-t0:.0f}s", flush=True)
        step = 4000
        tasks = [(lo, min(lo + step, a.height)) for lo in range(0, a.height, step)]
        ctx = get_context("fork")
        buf = {2: [], 3: []}
        with ctx.Pool(N_JOBS) as pool:
            for ti, res in enumerate(pool.imap_unordered(_topk_chunk, tasks)):
                if ti % 50 == 0:
                    print(f"   chunk {ti}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
                for src, rr, cc, dd, kk in res:
                    buf[src].append((rr, cc, dd, kk))
        a_ids = a["id"]
        for src, lst in buf.items():
            if not lst:
                continue
            rr = np.concatenate([x[0] for x in lst])
            cc = np.concatenate([x[1] for x in lst])
            dd = np.concatenate([x[2] for x in lst])
            kk = np.concatenate([x[3] for x in lst])
            out.append(pl.DataFrame({
                "s1_id": a_ids.gather(rr),
                "r_id": rk[src]["id"].gather(cc),
                "src": np.full(len(rr), src, dtype=np.int8),
                "bscore": dd,
                "brank": kk,
                "bself": selfscore[rr].astype(np.float32),
            }))
        print(f"[{split}:{c}] done {time.time()-t0:.0f}s", flush=True)
        _G.clear()
    cand = pl.concat(out)
    cand.write_parquet(work(f"{split}_blocks.parquet"))
    print(split, "candidates", cand.height, f"{time.time()-t0:.0f}s")
    return cand


if __name__ == "__main__":
    block_split(sys.argv[1])
