"""Normalise every record of one split (train/test) and write parquet files to WORK_DIR."""
import sys
from multiprocessing import get_context

import polars as pl

from config import N_JOBS, work
from io_utils import read_source
from textnorm import Normalizer

_N = None


def _init():
    global _N
    _N = Normalizer(work("translit.json"))


def _norm_chunk(args):
    names, addrs = args
    rows = [_N.name(n) + _N.address(a) for n, a in zip(names, addrs)]
    cols = list(zip(*rows))
    return pl.DataFrame({c: list(v) for c, v in zip(COLS, cols)}, schema=SCHEMA)


COLS = ["name", "core", "concat", "skel", "nflags",
        "addr", "alpha", "nums", "unums", "state", "aflags"]
SCHEMA = {c: (pl.Int8 if c in ("nflags", "aflags") else pl.Utf8) for c in COLS}


def normalize_df(df, pool, chunk=20000):
    tasks = ((df["business_name"][i:i + chunk].to_list(), df["business_address"][i:i + chunk].to_list())
             for i in range(0, df.height, chunk))
    out = pl.concat(list(pool.imap(_norm_chunk, tasks)))
    return pl.concat([df.select(pl.col("entity_id").alias("id"), "country"), out], how="horizontal")


def main(split):
    import os
    ctx = get_context("fork")
    with ctx.Pool(N_JOBS, initializer=_init) as pool:
        for k in (1, 2, 3):
            if os.path.exists(work(f"{split}_s{k}.parquet")) and os.environ.get("ER_REDO") != "1":
                continue
            df = read_source(split, k)
            out = normalize_df(df, pool)
            out.write_parquet(work(f"{split}_s{k}.parquet"))
            print(split, k, out.shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
