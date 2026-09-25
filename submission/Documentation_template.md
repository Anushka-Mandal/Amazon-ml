# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

The solution is a three-stage pipeline: blocking, pair classification, and assignment.
Blocking uses IDF-weighted sparse matching on phonetic name tokens, house numbers and address token bigrams, and keeps the top 12 candidates per source for each Source 1 entity.
A LightGBM classifier then scores each candidate pair using string-similarity features and "competition" features, which describe how a Source 2/3 record's score compares across all Source 1 entities that compete for it.
Finally, each Source 2/3 record is assigned to at most one Source 1 entity, and only links above a probability threshold chosen for macro F0.5 are kept.
On a held-out validation set of 150,000 Source 1 entities, the pipeline reaches **macro F0.5 = 0.965**, with pair-level precision of 0.994.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis of the training data gave these findings:

- **Size.** The training data has 2.2M Source 1 records and about 10.3M Source 2+3 records. The test data has 1.7M and about 10.0M. An all-pairs comparison is impossible, and the machine used had only 8 GB of RAM.
- **Cardinality.** Every Source 2/3 record matches at most one Source 1 entity: the 7.64M ground-truth links contain 7.64M distinct Source 2/3 IDs. About 74% of Source 2/3 records have a match, and the rest are distractors.
- **Cluster sizes.** A Source 1 entity has 0 to 11 matches, with a mean of 3.5. Only 5.6% of entities are singletons.
- **Countries.** Matches never cross country labels, so blocking runs inside each country label. The test set adds France, about 15% of test Source 1.
- **Source 1 is clean.** It is pure ASCII and title case. Sources 2 and 3 carry the noise:
  - legal-suffix changes, shuffles and additions (Pvt/Private, Ltd/(Limited), "Ltd Pvt")
  - injected honorifics or prefixes ("Mr", "Smt", ">>", "--")
  - character typos and leetspeak ("Abhi1asha", "5treet")
  - accent injection ("Límited")
  - web-style names ("horizonnutritionservices.com", "@bennettacademy")
  - DBA and completely different names, where only the address links the records
  - native-script names in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam and Odia. These are word-by-word transliterations of the Source 1 name.
- **Address noise.** Addresses show these patterns:
  - abbreviations (Rd/Road, St/Street, and even Street changed to "Saint")
  - US state names switched between full and 2-letter forms
  - Indian state names in native script
  - component reordering
  - dropped components, including the whole address for about 3.4% of records
  - extra or zero-padded numbers ("00156", "##21", "H.no")
  - unit, PMB and PO Box additions
  - truncated house numbers ("4106" becoming "410")
- **Repeated names.** Names are heavily reused inside Source 1: 846k Source 1 records share their exact name with another record, such as "Primary Care Group" 253 times. So the name alone is not enough, and address numbers and context decide most matches.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + constrained assignment (hybrid).  
**Core Innovation:** The core addition is global competition features taken from the blocking graph. For every candidate record, the model sees how this Source 1 entity's blocking score compares with the best other Source 1 entity competing for the same record. It also sees the record's rank among its Source 1 candidates. These features let a single tree model do most of the disambiguation between near-duplicate businesses. A one-to-many assignment step then enforces the observed cardinality: each Source 2/3 record gets at most one Source 1 entity.

The steps are:
1. **Learned transliteration** from training labels only. Native-script names are aligned token by token with their Source 1 name, which gives a dictionary of 1,347 tokens covering 96% of native-script tokens in test. Native-script address segments are mapped to the Source 1 segment they co-occur with, which recovers the 16 Indian state names.
2. **Normalisation** of names and addresses into canonical tokens (Section 4).
3. **Blocking** produces about 24 candidates per Source 1 entity (Section 3).
4. **Features and LightGBM** produce a match probability per candidate pair.
5. **Assignment and threshold**: each Source 2/3 record goes to its highest-probability Source 1 entity, and the link is kept only if p ≥ 0.70.

---

## 3. Candidate Generation (Blocking)

Every normalised record is turned into a set of **blocking items**:

| item type | example | purpose |
|---|---|---|
| `n` phonetic skeleton of each core-name token | lakshmi/laxmi → `lksm` | typo and transliteration tolerance |
| `nb` adjacent skeleton-token bigrams | `lksm_trdrs` | selective name evidence |
| `c` concatenated core name | `horizonnutritionservices` | web-style names |
| `d` address numbers (leading zeros stripped) | `8825` | house or plot numbers |
| `a` address word tokens | `winterbrook` | street and locality |
| `ab` adjacent address-token bigrams | `8825_winterbrook` | highly selective combination |

- **Weighting.** Each item is weighted by IDF computed on the Source 2+3 side, separately for each country label. Items with a document frequency above 3,000 are dropped, because they are too generic to discriminate and too costly to join.
- **Scoring.** For each country label, the Source 1 × Source k score is a sparse matrix product: the sum of the IDFs of the shared items. It is computed in parallel chunks of 4,000 Source 1 rows with SciPy.
- **Selection.** The top 12 candidates **per source** are kept for each Source 1 entity, so at most 24 in total. The limit exists because true matches per source number at most 5.
- **Open set of countries.** Country labels are handled as an open set: the loop runs over whatever labels exist. France went through the same code path without any special casing.

**Blocking keys used:** phonetic name tokens, name bigrams, concatenated name, address numbers, address tokens and address bigrams, with IDF weights.  
**Candidate pairs generated:**

| split | candidate pairs | pairs per S1 | reduction ratio vs. full cross product |
|---|---|---|---|
| train | 52,960,556 | 24.0 | 0.9999977 |
| test | 41,578,850 | 24.0 | 0.9999976 |

**How true matches were preserved:** Several independent key families are used, so one corrupted field does not remove a record: a typo in the street still leaves the number and the name, and a missing address still leaves the name items. The phonetic skeleton absorbs vowel typos and transliteration spelling. Learned transliteration puts native-script names into the same token space.

On train, blocking recall of ground-truth links is **0.940**:

| candidate rank within source | cumulative recall |
|---|---|
| < 1 | 0.445 |
| < 3 | 0.838 |
| < 5 | 0.909 |
| < 8 | 0.929 |
| < 12 | 0.940 |

Most missed links are records whose only remaining evidence is a generic, very frequent name. Examples are "Unified Diagnostic Services" with an empty address, or "Hotel Kanodia" with only the city. These could not be matched precisely anyway.

`candidate_pairs.tsv` is exactly this candidate set, which is what the model scores.

---

## 4. Matching Model

### Normalisation (`textnorm.py`)

**Names**
- Learned native-script token mapping, then unidecode folding for any remaining non-ASCII text.
- Lowercasing and folding "&" and "+" to "and".
- Web markers are stripped (`www.`, `.com`, `@`) and flagged.
- Acronyms are collapsed (L.L.C. → llc) and runs of single letters are merged.
- Legal forms are canonicalised: private/pvt, limited/ltd, corporation/corp, company/co/cie, and the French forms SAS, SARL, SASU, SCI and others.
- Common abbreviations are expanded or folded (svcs, assoc, intl, shri/sri/sree → shree).
- The *core name* removes legal forms, honorifics (mr, smt), country words (india, france) and stopwords.
- A phonetic skeleton is built: first letter kept, vowels dropped, ph→f, x→ks, c/q→k, z→s, w→v, h dropped, repeated letters collapsed.

**Addresses**
- Comma segments are mapped: native-script state to Latin name, then state or region names to a short code, which is kept as a separate field.
- Street-type abbreviations are folded to one form covering US, Indian and French usage (street/st/saint, road/rd, avenue/ave/av, boulevard/blvd/bd, rue/r, chemin/ch and so on).
- Ordinals are folded (1st, 1er, first → 1).
- Alternate city names are folded (Bombay/Mumbai, Madras/Chennai and similar).
- Filler tokens (no, h.no, door, null, n/a, #) are dropped.
- Alphanumeric tokens are split (5cnew → 5 cnew).
- Numbers are extracted with leading zeros stripped. They are split into house or plot numbers and unit or PO-box numbers.

### Features (65 in total)

**Name features**
- rapidfuzz ratio and token_set on the full normalised name.
- ratio, token_sort, token_set, partial_ratio and Jaro-Winkler on the core name.
- ratio and partial_ratio on the concatenated name.
- ratio, token_set and token_sort on the phonetic skeleton.
- Token counts and lengths.
- First-token equality.
- IDF-weighted overlap of skeleton tokens: shared weight, fraction of each side, max shared IDF and unshared weight.

**Address features**
- ratio, token_sort, token_set and partial_ratio on the normalised address.
- token_set on the address words.
- Number sets for house numbers and for all numbers: Jaccard, containment, first-number match, Levenshtein similarity of first numbers, counts, and ratio of the number strings.
- IDF-weighted address-token overlap.
- State equality, set to −1 when unknown.
- Missing-address and native-script flags.

**Blocking and context features**
- Blocking score, rank within source, the Source 1 entity's self-score, and score relative to the entity's best candidate and to its self-score.
- Competition features over the whole candidate graph:
  - the record's best score across all Source 1 entities
  - the ratio to that best score
  - the record's rank among its Source 1 candidates
  - the number of Source 1 entities competing for the record
  - **the margin between this Source 1 entity's score and the best other Source 1 entity's score for the same record**, which is the most important feature by gain
- Frequency of the Source 1 name and address within Source 1, which tells the model how ambiguous the name is.

No feature is a one-hot of country, so the model applies unchanged to the unseen France label.

**Model type:** LightGBM binary classifier (MIT licence). The settings are 127 leaves, learning rate 0.08, feature and bagging fraction 0.8, and early stopping on validation log-loss. Training stopped at iteration 1,171. It was trained on all candidate pairs of 300,000 randomly sampled training Source 1 entities, which is 7.2M pairs with 0.98M positives.

**Threshold selection method:** The decision rule was tuned directly on macro F0.5, computed exactly as in the challenge, on a separate validation sample of 150,000 training Source 1 entities. Ground-truth links lost in blocking count as misses, and singletons are included. The rule first keeps each Source 2/3 record only for its best Source 1 entity, then applies a probability threshold.

| threshold | 0.6 | 0.65 | **0.70** | 0.75 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| macro F0.5 | 0.9646 | 0.9650 | **0.9653** | 0.9653 | 0.9651 | 0.9633 |

The score curve is flat, so the result is not sensitive to the exact threshold. The non-exclusive variant and a relative-to-best threshold did not help.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation, 150k S1 entities:** **0.9653**

| slice | macro F0.5 |
|---|---|
| US | 0.9735 |
| India | 0.9530 |

| pair-level metric | value |
|---|---|
| precision | 0.9941 |
| recall | 0.9170 |
| singletons correctly left empty | 8,178 of 8,427 (97.0%) |

- **Common false positives (wrong merges):**
  - Near-identical sibling businesses: same generic name and same street, with the house number off by one or two ("6227" and "6228 Boundaries Rd").
  - Records with an empty address whose name exactly equals a Source 1 name.
  - Indian records reduced to "number + city" that collide with another company of the same name in the same city.
- **Common false negatives (missed matches):**
  - About 73% of missed links were never generated as candidates: generic name plus missing or city-only address.
  - The rest are scored below the threshold. Typical cases are heavy name corruption together with address truncation, DBA names that share nothing with the legal name, and house numbers corrupted in both sources.

---

## 6. Conclusion

A careful, data-driven normalisation stage works well for this problem. It combines a transliteration dictionary learned from the labels with abbreviation folding and phonetic skeletons. Paired with multi-key IDF blocking and a gradient-boosted classifier, it resolves business identities at 0.965 macro F0.5 with 99.4% pair precision. The main lesson is that context from the whole candidate graph matters more than any single string-similarity score when many businesses share generic names. Enforcing the one-record-to-one-entity structure is a cheap and effective precision guard. The largest remaining gain is blocking recall for records whose only evidence is a generic name.

---

## Appendix

### A. Code Artefacts

The code lives in `code/business_entity_resolution/`: all source is in `src/`, run instructions are in `README.md`, and pinned versions are in `requirements.txt`.
The entry point is `src/run_all.sh`, which runs `learn_translit.py`, `prepare.py train|test`, `blocking.py train`, `train.py`, `blocking.py test` and `predict.py`.
The last script writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Intermediate parquet files, the model file (`model.txt`) and the tuned decision rule (`decision.json`) go to `work/`.

**Compliance:**
- No external data, APIs or geocoding are used.
- The only domain knowledge is hand-written abbreviation, state-name and city-alias tables.
- The model is LightGBM (MIT) with far fewer than 8B parameters.
- Test data is used only transductively, for token IDF statistics and blocking, and never for labels.

### B. Additional Results

Top features by gain: b_margin_r, r_rank, anum_cont, anum_nb, b_rel_r, n_ratio, b_rel_self, c_partial, a_tset, n_tset, num_str_ratio, nm_idf_b.

Test-set outputs:

| country | test S1 entities | mean predicted links | predicted singletons |
|---|---|---|---|
| US | 663,106 | 3.30 | 5.8% |
| India | 809,986 | 3.11 | 6.7% |
| France (unseen in training) | 259,452 | 3.10 | 6.2% |
| **total** | **1,732,544** | **3.18** (5,510,652 links) | **6.3%** |

France was never seen during training, yet its link and singleton rates are close to those of the US and India. The singleton rate is also close to the 5.6% rate in the training data. This suggests the country-agnostic features transfer to the new label. Both output files pass `utils/validate_submission.py --check-ids`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
