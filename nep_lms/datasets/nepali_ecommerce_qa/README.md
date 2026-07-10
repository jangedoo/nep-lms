---
pretty_name: Nepali E-commerce Retrieval
task_categories:
- text-retrieval
language:
- ne
- en
tags:
- nepali
- ecommerce
- information-retrieval
- cross-lingual
- synthetic
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: row_id
    dtype: int64
  - name: document
    dtype: string
  - name: negative1
    dtype: string
  - name: negative2
    dtype: string
  - name: negative3
    dtype: string
  - name: query
    dtype: string
  - name: product_category
    dtype: string
  - name: language_mode
    dtype: string
  - name: query_intent
    dtype: string
  - name: doc_type
    dtype: string
  splits:
    - name: train
      num_examples: 1367
    - name: test
      num_examples: 425
    - name: valid
      num_examples: 198
---

# Nepali E-commerce Retrieval

This is a synthetic e-commerce retrieval dataset containing 1,990 query-positive pairs and three supplied negatives per query. Queries cover English, Devanagari Nepali, Romanized Nepali, and mixed modes; the documents are almost entirely English. The most accurate description is therefore **cross-lingual query-to-English retrieval**, not a fully multilingual document corpus.

> **Release status:** review required before publication. The data is synthetic/AI-generated, has not received a full human relevance audit, and does not yet have a verified license.

## Dataset structure

| Field | Description |
|---|---|
| `row_id` | Original integer row identifier; IDs are not contiguous. |
| `query` | E-commerce retrieval query. |
| `document` | Labeled positive document. |
| `negative1`–`negative3` | Supplied negative documents. |
| `product_category` | Product-domain label. |
| `language_mode` | Intended query-language style. |
| `query_intent` | Intended shopping or support intent. |
| `doc_type` | Document-format label. |

The release uses split names exactly `train`, `test`, and `valid`.

| split | rows | product_categories | language_modes | query_intents | document_types | Recall@1 | Recall@10 | MRR@10 | median_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1367 | 12 | 7 | 10 | 7 | 0.200 | 0.565 | 0.309 | 7 |
| test | 425 | 12 | 7 | 10 | 7 | 0.259 | 0.666 | 0.379 | 5 |
| valid | 198 | 12 | 7 | 10 | 7 | 0.182 | 0.566 | 0.293 | 6 |

## Major EDA findings

- **Rows and corpus:** 1,990 rows, 10 fields, and 7,959 unique candidate documents from 7,960 positive/negative slots.
- **Completeness:** no JSON errors, missing fields, nulls, or empty strings were found.
- **Identifier integrity:** `row_id` spans 0–1999 but omits 10 values: `[361, 438, 455, 549, 748, 1223, 1254, 1514, 1593, 1983]`.
- **Duplicate queries:** 22 rows repeat an exact query across 16 groups; normalized matching finds 24 duplicate rows. Repeated queries can link to different positive documents.
- **Label collision:** row `898` has a positive document duplicated in its negative set.
- **Scripts:** 1,530 queries are Latin-only, 250 are Devanagari-only, and 210 mix both. All 1,990 positive documents are Latin-only under this check.
- **Confounding:** every `language_mode` maps to exactly one `doc_type` and vice versa. Performance differences between these fields cannot be interpreted independently.
- **Synthetic design:** only 420 of 5,880 possible category/language/intent/document-type combinations occur.

### Composition

| dimension | values | minimum rows/value | maximum rows/value |
| --- | --- | --- | --- |
| product_category | 12 | 164 | 167 |
| language_mode | 7 | 283 | 285 |
| query_intent | 10 | 197 | 200 |
| doc_type | 7 | 283 | 285 |

### Text lengths

| field | min_words | median_words | mean_words | max_words | median_characters | mean_characters |
| --- | --- | --- | --- | --- | --- | --- |
| query | 4 | 12 | 12.9 | 33 | 71 | 73.9 |
| document | 38 | 68 | 69.6 | 129 | 414 | 420.8 |
| negative1 | 25 | 47 | 47.8 | 86 | 278 | 282.5 |
| negative2 | 19 | 42 | 43 | 84 | 247 | 252.2 |
| negative3 | 9 | 40 | 40.5 | 73 | 235 | 237.7 |

## E5 retrieval baseline

The baseline uses `intfloat/multilingual-e5-small` with the required `query:` and `passage:` prefixes and L2-normalized embeddings. Exact normalized duplicate queries treat all associated positive documents as relevant. The invalid four-document row is excluded only from the local four-way metric.

- Four-way positive-at-rank-1 accuracy: **91.75%** over 1,989 valid rows.
- Four-way MRR: **0.9571**.
- Full-corpus Recall@1 / Recall@5 / Recall@10: **0.2106 / 0.4719 / 0.5864**.
- Full-corpus MRR@10: **0.3222**; median positive rank: **6**.

### Full-corpus results by labeled language mode

| language_mode | rows | Recall@1 | Recall@5 | Recall@10 | MRR@10 | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| Romanized Nepali with English terms | 284 | 0.303 | 0.585 | 0.739 | 0.427 | 4 |
| English only | 285 | 0.263 | 0.572 | 0.681 | 0.389 | 4 |
| Romanized Nepali | 284 | 0.268 | 0.556 | 0.673 | 0.388 | 4.5 |
| English-Romanized Nepali | 283 | 0.166 | 0.473 | 0.583 | 0.300 | 7 |
| English-Nepali mixed | 285 | 0.232 | 0.477 | 0.575 | 0.336 | 6 |
| Trilingual mixed | 284 | 0.151 | 0.391 | 0.504 | 0.249 | 10 |
| Nepali (Devanagari) | 285 | 0.091 | 0.249 | 0.351 | 0.167 | 26 |

Because `language_mode` and `doc_type` are one-to-one confounded, this table must not be used to claim a purely linguistic performance difference.

## Split methodology

The split is designed so `valid` remains broadly representative while `test` is a semantic out-of-distribution holdout:

1. Queries are Unicode-NFKC normalized, case-folded, stripped of punctuation, and whitespace-normalized. Each duplicate-query group stays in one split.
2. Each row is represented by the equal-weight concatenation of its normalized E5 query and positive-document embeddings.
3. Query groups are clustered independently inside every product category with five-cluster K-means (`random_state=42`, `n_init=20`).
4. One 10–30%-sized, maximally isolated cluster per category is reserved for `test`. Consequently every product domain occurs in `test`, but its selected E5 clusters do not occur in `train` or `valid`.
5. `valid` targets 10% of all rows and is selected from the remainder by grouped candidate search, minimizing size and marginal-distribution error across category, language mode, intent, and document type. All other rows become `train`.

There is zero normalized-query overlap between splits. Every split contains all 12 categories, 7 language modes, 10 intents, and 7 document types.

The selected test clusters have mean within-category centroid cosine distance **0.037927** from the remainder, compared with **0.002932** for 2,000 size-matched random holdouts. The shift is **12.94× larger** (`p=0.000500`, one-sided permutation test), confirming that `test` is materially different in E5 representation space.

## Intended uses

- Contrastive retrieval or reranking experiments using supplied negatives.
- Cross-lingual Nepali/Romanized-Nepali query-to-English retrieval.
- Domain-generalization evaluation with the E5-clustered test split.
- Error analysis across shopping intents and product categories.

## Limitations and release recommendations

- The examples are synthetic/AI-generated and may contain unrealistic details or relevance errors.
- The dataset is small and has no completed human relevance assessment.
- The provided negatives are much easier than retrieval over the full corpus, so four-way accuracy alone overstates retrieval quality.
- Duplicate queries can have multiple linked positives and must be evaluated with multi-positive relevance.
- `language_mode` is inseparable from `doc_type` in the present design.
- Product and price claims should not be treated as current factual information.
- Confirm and add an appropriate license before public upload.

This dataset is best treated as a **synthetic training and evaluation seed**, not as production ground truth or a definitive Nepali retrieval benchmark without cleanup and human validation.
