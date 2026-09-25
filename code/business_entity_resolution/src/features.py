"""Pair features for candidate (Source 1 record, Source 2/3 record) pairs."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import work

REC_COLS = ["id", "country", "name", "core", "concat", "skel", "nflags",
            "addr", "alpha", "nums", "unums", "state", "aflags"]


def load_records(split):
    s1 = pl.read_parquet(work(f"{split}_s1.parquet"), columns=REC_COLS)
    r = pl.concat([pl.read_parquet(work(f"{split}_s{k}.parquet"), columns=REC_COLS)
                   for k in (2, 3)])
    return s1, r


def token_idf(s1, r):
    """IDF of skeleton name tokens and address alpha tokens, per country (whole split)."""
    out = {}
    for col, key in (("skel", "n"), ("alpha", "a")):
        t = (pl.concat([s1.select("country", col), r.select("country", col)])
             .with_columns(pl.col(col).str.split(" ")).explode(col)
             .filter(pl.col(col).str.len_chars() > 0)
             .group_by("country", col).len("df"))
        tot = pl.concat([s1.select("country"), r.select("country")]).group_by("country").len("N")
        t = t.join(tot, on="country").with_columns(
            (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias("idf")
        ).select("country", pl.col(col).alias("tok"), "idf")
        out[key] = t
    return out


def id_num(col, src=None):
    """Integer key for an entity id (prefix stripped); Source 2/3 ids are offset by source."""
    e = pl.col(col).str.slice(3).cast(pl.Int64)
    if src is not None:
        se = pl.col(src) if isinstance(src, str) else src
        e = e + se.cast(pl.Int64) * 10_000_000_000
    return e


def context_features(cand, s1):
    """Blocking-score context computed over the whole candidate set (integer keys s1n, rn)."""
    cand = cand.with_columns(
        pl.col("bscore").max().over("s1n", "src").alias("b_s1max"),
        pl.col("bscore").max().over("rn").alias("b_rmax"),
        pl.len().over("rn").cast(pl.Int16).alias("r_ncand"),
        pl.col("bscore").rank("ordinal", descending=True).over("rn").cast(pl.Int16).alias("r_rank"),
    )
    second = cand.filter(pl.col("r_rank") == 2).select("rn", pl.col("bscore").alias("b_r2"))
    cand = cand.join(second, on="rn", how="left").with_columns(pl.col("b_r2").fill_null(0.0))
    cand = cand.with_columns(
        (pl.col("bscore") / pl.col("b_s1max")).alias("b_rel_s1"),
        (pl.col("bscore") / pl.col("b_rmax")).alias("b_rel_r"),
        (pl.col("bscore") / pl.col("bself").clip(1e-3)).alias("b_rel_self"),
        # margin of this S1 over the best *other* S1 competing for the same record
        (pl.col("bscore") - pl.when(pl.col("r_rank") == 1).then(pl.col("b_r2"))
         .otherwise(pl.col("b_rmax"))).alias("b_margin_r"),
    ).drop("b_r2")
    freq = s1.select(
        id_num("id").alias("s1n"),
        pl.len().over("country", "core").cast(pl.Int32).alias("s1_name_freq"),
        pl.len().over("country", "addr").cast(pl.Int32).alias("s1_addr_freq"),
    )
    return cand.join(freq, on="s1n", how="left")


def _split_set(col):
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _idf_overlap(df, col_a, col_b, idf, name):
    """IDF-weighted overlap of two token columns; returns frac of A-weight, B-weight shared."""
    base = df.select("pid", "country", pl.col(col_a).alias("A"), pl.col(col_b).alias("B"))
    ea = (base.select("pid", "country", _split_set("A").list.unique().alias("tok")).explode("tok")
          .drop_nulls("tok").join(idf, on=["country", "tok"], how="left")
          .with_columns(pl.col("idf").fill_null(12.0)))
    eb = (base.select("pid", "country", _split_set("B").list.unique().alias("tok")).explode("tok")
          .drop_nulls("tok").join(idf, on=["country", "tok"], how="left")
          .with_columns(pl.col("idf").fill_null(12.0)))
    wa = ea.group_by("pid").agg(pl.col("idf").sum().alias("wa"))
    wb = eb.group_by("pid").agg(pl.col("idf").sum().alias("wb"))
    sh = ea.join(eb.select("pid", "tok"), on=["pid", "tok"]).group_by("pid").agg(
        pl.col("idf").sum().alias("ws"), pl.col("idf").max().alias("wmax"))
    res = (df.select("pid").join(wa, on="pid", how="left").join(wb, on="pid", how="left")
           .join(sh, on="pid", how="left").fill_null(0.0).sort("pid"))
    wa_, wb_, ws_ = res["wa"].to_numpy(), res["wb"].to_numpy(), res["ws"].to_numpy()
    return {
        f"{name}_idf_a": ws_ / np.maximum(wa_, 1e-6),
        f"{name}_idf_b": ws_ / np.maximum(wb_, 1e-6),
        f"{name}_idf_shared": ws_,
        f"{name}_idf_max": res["wmax"].to_numpy(),
        f"{name}_idf_unshared": (wa_ + wb_ - 2 * ws_),
    }


def _sim(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, **kw).astype(np.float32)


def _set_feats(a_list, b_list, prefix):
    """Set features between two space-separated number strings."""
    n = len(a_list)
    jac = np.zeros(n, np.float32)
    cont = np.zeros(n, np.float32)
    first = np.zeros(n, np.float32)
    firstlev = np.zeros(n, np.float32)
    na = np.zeros(n, np.float32)
    nb = np.zeros(n, np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        sa = a.split()
        sb = b.split()
        na[i], nb[i] = len(sa), len(sb)
        if not sa or not sb:
            jac[i] = cont[i] = first[i] = firstlev[i] = -1
            continue
        A, B = set(sa), set(sb)
        inter = len(A & B)
        jac[i] = inter / len(A | B)
        cont[i] = inter / min(len(A), len(B))
        first[i] = 1.0 if sa[0] in B or sb[0] in A else 0.0
        firstlev[i] = Levenshtein.normalized_similarity(sa[0], sb[0])
    return {f"{prefix}_jac": jac, f"{prefix}_cont": cont, f"{prefix}_first": first,
            f"{prefix}_firstlev": firstlev, f"{prefix}_na": na, f"{prefix}_nb": nb}


def pair_features(pairs, s1, r, idf):
    """pairs: DataFrame with s1_id, r_id + context columns. Returns pairs with features."""
    p = pairs.with_row_index("pid")
    a = s1.rename({c: c + "_a" for c in s1.columns if c != "id"}).rename({"id": "s1_id"})
    b = r.rename({c: c + "_b" for c in r.columns if c != "id"}).rename({"id": "r_id"})
    d = (p.select("pid", "s1_id", "r_id").join(a, on="s1_id", how="left")
         .join(b, on="r_id", how="left").sort("pid"))
    f = {}
    L = lambda c: d[c].fill_null("").to_list()
    name_a, name_b = L("name_a"), L("name_b")
    core_a, core_b = L("core_a"), L("core_b")
    f["n_ratio"] = _sim(name_a, name_b, fuzz.ratio)
    f["n_tset"] = _sim(name_a, name_b, fuzz.token_set_ratio)
    f["c_ratio"] = _sim(core_a, core_b, fuzz.ratio)
    f["c_tsort"] = _sim(core_a, core_b, fuzz.token_sort_ratio)
    f["c_tset"] = _sim(core_a, core_b, fuzz.token_set_ratio)
    f["c_partial"] = _sim(core_a, core_b, fuzz.partial_ratio)
    f["c_jw"] = _sim(core_a, core_b, JaroWinkler.normalized_similarity)
    del name_a, name_b
    cc_a, cc_b = L("concat_a"), L("concat_b")
    f["cc_ratio"] = _sim(cc_a, cc_b, fuzz.ratio)
    f["cc_partial"] = _sim(cc_a, cc_b, fuzz.partial_ratio)
    del cc_a, cc_b
    sk_a, sk_b = L("skel_a"), L("skel_b")
    f["sk_ratio"] = _sim(sk_a, sk_b, fuzz.ratio)
    f["sk_tset"] = _sim(sk_a, sk_b, fuzz.token_set_ratio)
    f["sk_tsort"] = _sim(sk_a, sk_b, fuzz.token_sort_ratio)
    f["c_ntok_a"] = np.array([len(x.split()) for x in core_a], np.float32)
    f["c_ntok_b"] = np.array([len(x.split()) for x in core_b], np.float32)
    f["c_len_a"] = np.array([len(x) for x in core_a], np.float32)
    f["c_len_b"] = np.array([len(x) for x in core_b], np.float32)
    f["first_tok_eq"] = np.array(
        [1.0 if (x.split()[:1] == y.split()[:1] and x) else 0.0 for x, y in zip(sk_a, sk_b)],
        np.float32)
    del core_a, core_b, sk_a, sk_b
    ad_a, ad_b = L("addr_a"), L("addr_b")
    f["a_ratio"] = _sim(ad_a, ad_b, fuzz.ratio)
    f["a_tsort"] = _sim(ad_a, ad_b, fuzz.token_sort_ratio)
    f["a_tset"] = _sim(ad_a, ad_b, fuzz.token_set_ratio)
    f["a_partial"] = _sim(ad_a, ad_b, fuzz.partial_ratio)
    f["a_len_a"] = np.array([len(x) for x in ad_a], np.float32)
    f["a_len_b"] = np.array([len(x) for x in ad_b], np.float32)
    del ad_a, ad_b
    al_a, al_b = L("alpha_a"), L("alpha_b")
    f["al_tset"] = _sim(al_a, al_b, fuzz.token_set_ratio)
    del al_a, al_b
    na_, nb_ = L("nums_a"), L("nums_b")
    f.update(_set_feats(na_, nb_, "num"))
    ua, ub = L("unums_a"), L("unums_b")
    f.update(_set_feats([x + " " + y for x, y in zip(na_, ua)],
                        [x + " " + y for x, y in zip(nb_, ub)], "anum"))
    f["num_str_ratio"] = _sim(na_, nb_, fuzz.ratio)
    del na_, nb_, ua, ub
    st_a, st_b = d["state_a"].fill_null(""), d["state_b"].fill_null("")
    f["state_eq"] = np.where((st_a == "") | (st_b == ""), -1,
                             (st_a == st_b).cast(pl.Int8).to_numpy()).astype(np.float32)
    f["nflags_b"] = d["nflags_b"].fill_null(0).to_numpy().astype(np.float32)
    f["aflags_b"] = d["aflags_b"].fill_null(0).to_numpy().astype(np.float32)
    f["aflags_a"] = d["aflags_a"].fill_null(0).to_numpy().astype(np.float32)
    f.update(_idf_overlap(d.select("pid", pl.col("country_a").alias("country"), "skel_a",
                                   "skel_b"), "skel_a", "skel_b", idf["n"], "nm"))
    f.update(_idf_overlap(d.select("pid", pl.col("country_a").alias("country"), "alpha_a",
                                   "alpha_b"), "alpha_a", "alpha_b", idf["a"], "ad"))
    feats = pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    return pl.concat([pairs, feats], how="horizontal")


CONTEXT_COLS = ["src", "bscore", "brank", "bself", "b_s1max", "b_rmax", "r_ncand", "r_rank",
                "b_rel_s1", "b_rel_r", "b_rel_self", "b_margin_r", "s1_name_freq",
                "s1_addr_freq"]


def feature_columns(df):
    skip = {"s1_id", "r_id", "label", "pid", "country"}
    return [c for c in df.columns if c not in skip]
