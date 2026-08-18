# Gutenberg surprisal benchmark

This directory implements both the engineering benchmark and the resumable
full-corpus Gutenberg spectrum-validation pipeline.  Benchmark diagnostics are
not confirmatory paper results.

## Fixed design

- Languages: `en fr de it es pt fi nl hu zh`, balanced within each manifest.
- Raw-text anchors: nearest paragraph to document fractions `1/6, 1/2, 5/6`.
- Per anchor: 1024 context tokens followed by 4096 scored target tokens.
- Checkpoints: pretrained/base BF16 only; no chat template or instruction tags.
- The same raw-text anchors are used for the Llama and Qwen tokenizers.
- Ten octave bands cover periods `(1024,2048]` through `(2,4]` tokens.
- Three windows are averaged before a document enters a statistical summary.

## Server paths

```text
Corpus       /nas/data/gutenberg/corpus
Llama        /nas/model/Llama3.1/Llama3.1-8B
Qwen         /nas/model/qwen3/Qwen3-8B-Base
Environment  /home/xy/miniconda3/envs/cs310
Results      results/gutenberg_benchmark
```

## Pipeline

1. `prepare_gutenberg_windows.py` tokenizes complete documents, aligns the
   three raw-text anchors, validates non-overlap, and writes fixed input arrays.
   A skipped document is committed transactionally: none of its windows can
   leak into the output.
2. `prepare_gutenberg_chunks.py` splits a full manifest (4096 documents by
   default), prepares each tokenizer-specific chunk, validates it, and writes
   a `chunk-NNNNN._SUCCESS.json` checkpoint.  Re-running resumes completed
   chunks and produces eligible/skipped manifests.
3. `score_gutenberg_windows.py` computes float32 token NLL from BF16 model
   logits.  It supports one GPU or a rank within data-parallel execution and
   atomically resumes a previously validated rank shard.
4. `run_8gpu_score.sh` launches eight independent model replicas, one per GPU,
   validates every rank, and seals the chunk with `_SUCCESS.json`.
5. `run_chunked_8gpu_score.sh` processes all prepared chunks in order.  A
   restart skips sealed chunks and completed rank shards.
6. `analyze_gutenberg_benchmark.py` runs 200 within-window permutations and
   writes window, document, and language-band diagnostics.

Each score rank writes one NPZ shard and one JSON summary.  Document IDs and
all window alignment metadata are retained so runs can be audited or resumed.

## Full-corpus commands

The examples below use the NAS output root created for the full experiment.
The manifest must contain unique `document_id` values and a `language` column.

The tracked full manifest is built deterministically from the 2026-07-26
Project Gutenberg catalog snapshot and the numeric `.txt` files actually
present in the corpus:

```bash
python build_gutenberg_manifest.py \
  --catalog /private/tmp/pg_catalog_2026-07-26.csv.gz \
  --catalog-snapshot-date 2026-07-26 \
  --corpus-dir /Users/xy/projects/gutenberg/corpus \
  --output manifests/gutenberg_full.csv
```

`gutenberg_full.meta.json` records the catalog, corpus-inventory, and manifest
SHA-256 hashes plus language-label and catalog-type counts.  Language labels,
including multi-language values such as `de; en`, are retained verbatim.  The
manifest includes all locally present catalog types rather than silently
restricting the cohort to `Text`.

```bash
python prepare_gutenberg_chunks.py \
  --manifest manifests/gutenberg_full.csv \
  --corpus-dir /nas/data/gutenberg/corpus \
  --tokenizer /nas/model/Llama3.1/Llama3.1-8B \
  --tokenizer-label llama31 \
  --output-root /nas/xy/gutenberg_surprisal

./run_chunked_8gpu_score.sh \
  /nas/xy/gutenberg_surprisal/prepared/llama31 \
  /nas/model/Llama3.1/Llama3.1-8B \
  llama31 \
  /nas/xy/gutenberg_surprisal/surprisal/llama31 \
  4
```

Use the corresponding Qwen paths and label `qwen3` for the second model.
Do not manually create `_SUCCESS.json`: it is written only after array shape,
finite-value, metadata, model, rank, and world-size validation succeeds.
