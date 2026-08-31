# Gutenberg cross-language mean spectra

This directory contains document-weighted average surprisal spectra for the
ten focal Gutenberg languages, evaluated with Llama-3.1-8B and Qwen3-8B-Base.

## Aggregation

1. Each 4,096-token surprisal window is standardized independently.
2. Its positive-frequency periodogram is normalized to unit total power.
3. Per-bin power is multiplied by 2,048, so a flat white-noise spectrum has a
   relative spectral density near 1.
4. The three windows are averaged within a document.
5. Document spectra are averaged within each language; ribbons in the main
   figure are normal-theory 95% confidence intervals across documents.
6. Publication plots show only the validated range $2 < T \leq 2048$ model
   tokens, omitting the one-cycle and Nyquist bins.

## Methods compared

- **Exact-bin mean:** no smoothing; one value for every Fourier bin.
- **Log-period bins:** exact-bin densities averaged within 96 equally spaced
  bins on the log2-period axis.  This is the recommended publication curve.
- **Savitzky--Golay:** a third-order, 15-point filter applied to the log-binned
  log density.
- **Smoothing spline:** a penalized cubic spline fitted to the log-binned log
  density, serving as a lightweight analogue of the earlier GAM plots.

The three processed curves are visually very similar.  Relative to the direct
log-bin curve, the root-mean-square log deviations are 0.011 (Savitzky--Golay)
and 0.025 (spline) for English, and 0.017 and 0.032 for Chinese.  A GAM over
millions of raw frequency observations is therefore unnecessary.

Across the validated log-period bins, Llama--Qwen Pearson correlations range
from 0.987 to 0.9998 across the ten languages.  All languages show increasing
relative spectral density at long periods.  Chinese has the largest long-period
increase and the largest evaluator-model difference.

## Files

- `gutenberg_language_spectra.{pdf,png}`: recommended ten-language figure.
- `gutenberg_language_spectra_linear_x.{pdf,png}`: the same estimates with a
  linear rather than log2 period axis.
- `gutenberg_spectrum_method_comparison.{pdf,png}`: exact-bin, log-bin,
  Savitzky--Golay, and spline comparison for English and Chinese.
- `language_spectra_exact.csv`: exact Fourier-bin means and standard errors.
- `language_spectra_logbin.csv`: log-period-bin means and standard errors.
- `language_spectra_summary.json`: model, sample-size, and runtime audit.

The server aggregation took 46.9 seconds.  The source NLL shards remain under
`/nas/xy/gutenberg_surprisal/surprisal`; the server-side summaries are under
`/nas/xy/gutenberg_surprisal/figures/language_spectra`.
