# MDPI Entropy experiments

Reproducible experiments for the spectrum and periodicity manuscript.

The first pilot re-evaluates the Brown and WSJ surprisal sequences using document-level paired comparisons against repeated within-document shuffle surrogates.

## Brown/WSJ pilot

The inputs are the existing `brown_nlls_doc.pkl` and `wsj_nlls_doc.pkl` files in `period-playground/cl`. They contain token surprisals computed with Llama-3-8B-base.

```bash
MPLCONFIGDIR=/tmp/mdpi-entropy-mpl \
  python run_spectrum_pilot.py
```

The default run uses the first 1024 tokens of every eligible document, 200 independent within-document shuffles, eight octave-spaced period bands, document-level paired effects, an empirical omnibus test, bootstrap confidence intervals, and Benjamini--Hochberg correction across the 16 corpus-by-band tests.

Outputs are written to `results/pilot_brown_wsj/`.

## Gutenberg spectrum validation

The resumable Gutenberg pipeline, deterministic full-corpus manifest, fixed
100/800-document benchmark manifests, and measured benchmark report are under
`gutenberg_benchmark/`.  Large prepared arrays, token-level surprisal shards,
logs, and diagnostic outputs are intentionally excluded from Git.
