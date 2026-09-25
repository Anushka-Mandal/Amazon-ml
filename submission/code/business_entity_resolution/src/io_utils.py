"""TSV reading helpers (explicit tab separator, no quoting, everything as string)."""
import polars as pl

from config import gt_path, src_path


def read_source(split, k):
    df = pl.read_csv(
        src_path(split, k), separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True,
    )
    return df.with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    )


def read_ground_truth():
    gt = pl.read_csv(gt_path(), separator="\t", quote_char=None, infer_schema=False,
                     missing_utf8_is_empty_string=True)
    return (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"})
    )
