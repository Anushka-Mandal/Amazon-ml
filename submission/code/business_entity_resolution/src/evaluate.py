"""Macro F0.5 exactly as described in the challenge (singletons included)."""
import polars as pl


def macro_f05(pred, truth, s1_ids, beta=0.5):
    """pred / truth: DataFrames (s1_id, r_id) of predicted / true links; s1_ids: all S1 ids."""
    b2 = beta * beta
    base = pl.DataFrame({"s1_id": s1_ids})
    npred = pred.group_by("s1_id").len("np")
    ntrue = truth.group_by("s1_id").len("nt")
    tp = pred.join(truth, on=["s1_id", "r_id"]).group_by("s1_id").len("tp")
    d = (base.join(npred, on="s1_id", how="left").join(ntrue, on="s1_id", how="left")
         .join(tp, on="s1_id", how="left").fill_null(0))
    d = d.with_columns(
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
        .when((pl.col("tp") == 0)).then(0.0)
        .otherwise(
            (1 + b2) * (pl.col("tp") / pl.col("np")) * (pl.col("tp") / pl.col("nt"))
            / (b2 * pl.col("tp") / pl.col("np") + pl.col("tp") / pl.col("nt"))
        ).alias("f")
    )
    return d["f"].mean()
