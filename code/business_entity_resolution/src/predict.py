"""Score every test candidate pair and write output/matching_results.tsv + candidate_pairs.tsv."""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from config import OUT_DIR, work
from pipeline_common import build_features, decide, write_outputs


def main():
    cfg = json.load(open(work("decision.json")))
    model = lgb.Booster(model_file=work("model.txt"))
    cols = cfg["features"]
    files = build_features("test", None, tag="all")
    scored = []
    for fn in files:
        df = pl.read_parquet(fn)
        p = model.predict(df.select(cols).to_numpy())
        scored.append(df.select("s1_id", "r_id").with_columns(pl.Series("p", p.astype(np.float32))))
    scored = pl.concat(scored)
    scored.write_parquet(work("test_scored.parquet"))
    links = decide(scored, cfg["t_abs"], cfg["t_rel"], cfg["exclusive"])
    s1_ids = pl.read_parquet(work("test_s1.parquet"), columns=["id"])["id"]
    m, c = write_outputs(s1_ids, scored, links, OUT_DIR)
    print("test S1:", len(s1_ids), "links:", links.height,
          "singletons predicted:", (m["matched_entity_ids"] == "").sum(),
          "candidate pairs:", scored.height)


if __name__ == "__main__":
    main()
