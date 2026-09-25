# Business Entity Resolution: reproducible pipeline

Blocking + gradient-boosted pair classifier + one-to-many assignment, tuned for macro F0.5.
It uses only the provided training and test files. There are no external lookups, APIs or pretrained models.

## Environment

- Python 3.10+ (developed on 3.14, macOS arm64, 8 cores, 8 GB RAM)
- `pip install -r requirements.txt`
- The model is LightGBM (MIT licence). It is a gradient-boosted tree model with far fewer than 8B parameters.

## Data layout

The code expects the challenge folder layout:

```
<ROOT>/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
<ROOT>/dataset/test/test_source{1,2,3}.tsv
```

`<ROOT>` defaults to three levels above `src/`, which is the `student_resource/` folder when this
package sits at `student_resource/code/business_entity_resolution`. You can override the paths with environment
variables:

| variable   | meaning                                   | default           |
|------------|-------------------------------------------|-------------------|
| `ER_ROOT`  | project root                              | see above         |
| `ER_DATA`  | dataset folder                            | `$ER_ROOT/dataset`|
| `ER_WORK`  | intermediate files (parquet, model)       | `$ER_ROOT/work`   |
| `ER_OUT`   | output folder for the two TSVs            | `$ER_ROOT/output` |
| `ER_JOBS`  | worker processes                          | CPU count         |

## Run end to end

```bash
cd src
PYTHON=python3 ./run_all.sh
```

The script runs these steps in order:

| step | script | what it does | approx. time (8 cores) |
|------|--------|--------------|------------------------|
| 1 | `learn_translit.py` | learns native-script to Latin dictionaries from the training labels | 1 min |
| 2 | `prepare.py train` / `prepare.py test` | normalises names and addresses into parquet | 2 min each |
| 3 | `blocking.py train` | builds the candidate set for train | 25 min |
| 4 | `train.py` | builds features for 300k train and 150k validation S1 entities, trains LightGBM, tunes the threshold | 20 min |
| 5 | `blocking.py test` | builds the candidate set for test | 25 min |
| 6 | `predict.py` | scores all test candidates and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` | 40 min |

Then validate the outputs:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source files

| file | role |
|------|------|
| `config.py` | paths and settings |
| `io_utils.py` | TSV readers (explicit tab separator, no quoting) |
| `learn_translit.py` | learns the Indic-script token and address-segment dictionaries |
| `textnorm.py` | name and address canonicalisation, phonetic skeleton, number extraction |
| `prepare.py` | parallel normalisation of all sources |
| `blocking.py` | IDF-weighted sparse item matching per country and top-K candidates per source |
| `features.py` | pair features (rapidfuzz similarities, number sets, IDF overlap) and competition features |
| `pipeline_common.py` | chunked feature building, decision rule, TSV writers |
| `evaluate.py` | macro F0.5 exactly as defined by the challenge |
| `train.py` | model training and threshold tuning |
| `predict.py` | test inference and output files |

`candidate_pairs.tsv` contains exactly the pairs that the model scores. `matching_results.tsv` is a subset of it.
