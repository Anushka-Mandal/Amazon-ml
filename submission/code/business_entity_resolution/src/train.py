"""Train the pair classifier on the training split and tune the decision rule for macro F0.5."""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from config import work
from evaluate import macro_f05
from features import feature_columns
from io_utils import read_ground_truth
from pipeline_common import build_features, decide

N_TRAIN_S1 = 300000
N_VALID_S1 = 150000
SEED = 13


def load(files, gt):
    df = pl.concat([pl.read_parquet(f) for f in files])
    return df.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")),
                   on=["s1_id", "r_id"], how="left").with_columns(pl.col("label").fill_null(0))


def main():
    gt = read_ground_truth()
    s1_all = pl.read_parquet(work("train_s1.parquet"), columns=["id"])["id"]
    cand = pl.read_parquet(work("train_blocks.parquet"), columns=["s1_id", "r_id", "src", "brank"])
    hit = cand.join(gt, on=["s1_id", "r_id"])
    print(f"blocking: {cand.height} pairs, {cand.height / len(s1_all):.1f}/S1, "
          f"recall {hit.height / gt.height:.4f}")
    for k in (1, 2, 3, 5, 8, 12):
        print(f"   recall@rank<{k}: {hit.filter(pl.col('brank') < k).height / gt.height:.4f}")
    del cand, hit

    perm = s1_all.sample(fraction=1.0, shuffle=True, seed=SEED)
    tr_ids = perm[:N_TRAIN_S1]
    va_ids = perm[N_TRAIN_S1:N_TRAIN_S1 + N_VALID_S1]
    tr_files = build_features("train", tr_ids, tag="tr")
    va_files = build_features("train", va_ids, tag="va")

    tr = load(tr_files, gt)
    va = load(va_files, gt)
    cols = feature_columns(tr.drop("label"))
    print("features:", len(cols), "train pairs", tr.height, "pos", tr["label"].sum(),
          "valid pairs", va.height)
    dtr = lgb.Dataset(tr.select(cols).to_numpy(), tr["label"].to_numpy(), feature_name=cols,
                      free_raw_data=True)
    dva = lgb.Dataset(va.select(cols).to_numpy(), va["label"].to_numpy(), reference=dtr)
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=8, verbose=-1, max_bin=255)
    model = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(work("model.txt"))
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(c, round(g / 1e3)) for c, g in imp[:20]])

    va = va.with_columns(pl.Series("p", model.predict(va.select(cols).to_numpy())))
    va.select("s1_id", "r_id", "label", "p").write_parquet(work("valid_scored.parquet"))
    truth = gt.join(pl.DataFrame({"s1_id": va_ids}), on="s1_id", how="semi")
    best = (0, None)
    for ex in (True, False):
        for t in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
            for tr_ in (0.0, 0.3, 0.5):
                links = decide(va.select("s1_id", "r_id", "p"), t, tr_, ex)
                f = macro_f05(links, truth, va_ids)
                if f > best[0]:
                    best = (f, dict(t_abs=t, t_rel=tr_, exclusive=ex))
        print("exclusive", ex, "best so far", best)
    print("VALID macro F0.5:", best)
    with open(work("decision.json"), "w") as fh:
        json.dump(dict(best[1], valid_f05=best[0], features=cols,
                       best_iteration=model.best_iteration), fh, indent=1)


if __name__ == "__main__":
    main()
