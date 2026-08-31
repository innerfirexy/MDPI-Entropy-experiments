#!/usr/bin/env python3
"""Aggregate and plot document-weighted Gutenberg spectra by language.

The scorer stores three 4,096-token surprisal windows per document.  This
script computes a normalized periodogram for each window, averages the three
windows within a document, and only then averages documents within a language.
It writes exact Fourier-bin and log-period-bin summaries and compares several
lightweight smoothers on the already aggregated curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


DEFAULT_LANGUAGES = ("en", "fr", "de", "it", "es", "pt", "fi", "nl", "hu", "zh")
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "fi": "Finnish",
    "nl": "Dutch",
    "hu": "Hungarian",
    "zh": "Chinese",
}
MODEL_NAMES = {"llama31": "Llama-3.1-8B", "qwen3": "Qwen3-8B-Base"}
MODEL_COLORS = {"llama31": "#0072B2", "qwen3": "#D55E00"}


@dataclass
class RunningMoments:
    count: int
    total: np.ndarray
    total_sq: np.ndarray

    @classmethod
    def zeros(cls, width: int) -> "RunningMoments":
        return cls(0, np.zeros(width, dtype=np.float64), np.zeros(width, dtype=np.float64))

    def update(self, values: np.ndarray) -> None:
        self.count += 1
        self.total += values
        self.total_sq += values * values

    def mean_and_se(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.total / self.count
        if self.count < 2:
            return mean, np.full_like(mean, np.nan)
        variance = (self.total_sq - self.count * mean * mean) / (self.count - 1)
        return mean, np.sqrt(np.maximum(variance, 0.0) / self.count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["llama31", "qwen3"])
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--log-bins", type=int, default=96)
    parser.add_argument("--fft-batch-size", type=int, default=256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only and args.input_root is None:
        parser.error("--input-root is required unless --plot-only is used")
    return args


def normalized_relative_density(values: np.ndarray) -> np.ndarray:
    """Return normalized positive-frequency power relative to white noise."""
    values = np.asarray(values, dtype=np.float64)
    std = values.std(axis=1, keepdims=True)
    if np.any(std == 0):
        raise ValueError("A surprisal window has zero variance")
    standardized = (values - values.mean(axis=1, keepdims=True)) / std
    power = np.abs(np.fft.rfft(standardized, axis=1)) ** 2
    power[:, 0] = 0.0
    power /= np.maximum(power.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    positive = power[:, 1:]
    return positive * positive.shape[1]


def make_log_period_bins(length: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fourier_k = np.arange(1, length // 2 + 1)
    periods = length / fourier_k
    edges = np.linspace(math.log2(2.0), math.log2(float(length)), count + 1)
    assignments = np.clip(np.digitize(np.log2(periods), edges) - 1, 0, count - 1)
    centers = 2.0 ** ((edges[:-1] + edges[1:]) / 2.0)
    bin_counts = np.bincount(assignments, minlength=count)
    return assignments, centers, bin_counts


def bin_density(values: np.ndarray, assignments: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    totals = np.bincount(assignments, weights=values, minlength=len(bin_counts))
    result = np.full(len(bin_counts), np.nan)
    nonempty = bin_counts > 0
    result[nonempty] = totals[nonempty] / bin_counts[nonempty]
    return result


def shard_paths(input_root: Path, model: str) -> list[Path]:
    paths = sorted((input_root / model).glob("chunk-*/*.npz"))
    if not paths:
        paths = sorted((input_root / model).glob(f"{model}.rank*-of-*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ shards found for {model} under {input_root}")
    return paths


def aggregate_model(
    input_root: Path,
    model: str,
    languages: set[str],
    log_bins: int,
    fft_batch_size: int,
) -> tuple[dict[str, RunningMoments], dict[str, RunningMoments], np.ndarray, np.ndarray, np.ndarray]:
    exact: dict[str, RunningMoments] = {}
    binned: dict[str, RunningMoments] = {}
    length = 0
    periods = np.array([])
    assignments = centers = bin_counts = np.array([])

    for path in shard_paths(input_root, model):
        with np.load(path, allow_pickle=False) as shard:
            nll = shard["nll"]
            document_ids = shard["document_id"]
            shard_languages = shard["language"].astype(str)
            selected = np.isin(shard_languages, list(languages))
            if not np.any(selected):
                continue
            nll = nll[selected]
            document_ids = document_ids[selected]
            shard_languages = shard_languages[selected]

        if length == 0:
            length = int(nll.shape[1])
            periods = length / np.arange(1, length // 2 + 1)
            assignments, centers, bin_counts = make_log_period_bins(length, log_bins)

        spectra = []
        for start in range(0, len(nll), fft_batch_size):
            spectra.append(normalized_relative_density(nll[start : start + fft_batch_size]))
        spectra = np.concatenate(spectra)

        order = np.argsort(document_ids, kind="stable")
        document_ids = document_ids[order]
        shard_languages = shard_languages[order]
        spectra = spectra[order]
        unique_ids, starts, counts = np.unique(document_ids, return_index=True, return_counts=True)
        if np.any(counts != 3):
            bad = unique_ids[counts != 3][:5]
            raise AssertionError(f"Expected three windows per document in {path}; examples: {bad}")

        for start in starts:
            language = str(shard_languages[start])
            if not np.all(shard_languages[start : start + 3] == language):
                raise AssertionError(f"Language mismatch within a document in {path}")
            document_spectrum = spectra[start : start + 3].mean(axis=0)
            exact.setdefault(language, RunningMoments.zeros(len(periods))).update(document_spectrum)
            document_binned = bin_density(document_spectrum, assignments, bin_counts)
            populated = bin_counts > 0
            binned.setdefault(language, RunningMoments.zeros(int(populated.sum()))).update(
                document_binned[populated]
            )

    return exact, binned, periods, centers[bin_counts > 0], bin_counts[bin_counts > 0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_plot_summaries(
    output_dir: Path,
    models: list[str],
    languages: list[str],
) -> tuple[
    dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]],
    dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]],
    dict[str, np.ndarray],
    np.ndarray,
]:
    exact_values: dict[tuple[str, str], list[tuple[int, float, float, float]]] = defaultdict(list)
    with (output_dir / "language_spectra_exact.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["language"])
            exact_values[key].append((
                int(row["fourier_k"]), float(row["period"]),
                float(row["mean_relative_density"]), float(row["se_relative_density"]),
            ))

    binned_values: dict[tuple[str, str], list[tuple[int, float, float, float]]] = defaultdict(list)
    with (output_dir / "language_spectra_logbin.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["language"])
            binned_values[key].append((
                int(row["log_bin"]), float(row["period_center"]),
                float(row["mean_relative_density"]), float(row["se_relative_density"]),
            ))

    exact_summaries = {model: {} for model in models}
    binned_summaries = {model: {} for model in models}
    model_periods = {}
    binned_periods = None
    summary = json.loads((output_dir / "language_spectra_summary.json").read_text(encoding="utf-8"))
    document_counts = summary["document_counts"]
    for model in models:
        for language in languages:
            key = (model, language)
            if key not in exact_values:
                continue
            exact_rows = sorted(exact_values[key])
            binned_rows = sorted(binned_values[key])
            exact_period = np.asarray([row[1] for row in exact_rows])
            bin_period = np.asarray([row[1] for row in binned_rows])
            model_periods[model] = exact_period
            binned_periods = bin_period
            documents = int(document_counts[model][language])
            exact_summaries[model][language] = (
                np.asarray([row[2] for row in exact_rows]),
                np.asarray([row[3] for row in exact_rows]),
                documents,
            )
            binned_summaries[model][language] = (
                np.asarray([row[2] for row in binned_rows]),
                np.asarray([row[3] for row in binned_rows]),
                documents,
            )
    if binned_periods is None:
        raise ValueError(f"No requested model-language rows in {output_dir}")
    return exact_summaries, binned_summaries, model_periods, binned_periods


def plot_language_facets(
    output: Path,
    summaries: dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]],
    periods: np.ndarray,
    languages: list[str],
) -> None:
    import matplotlib.pyplot as plt

    display = (periods > 2.0) & (periods <= 2048.0)
    display_periods = periods[display]
    fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.6), sharex=True, sharey=True)
    for axis, language in zip(axes.flat, languages):
        counts = []
        for model, by_language in summaries.items():
            if language not in by_language:
                continue
            mean, se, documents = by_language[language]
            counts.append(documents)
            mean = mean[display]
            se = se[display]
            axis.plot(display_periods, mean, color=MODEL_COLORS.get(model), linewidth=1.35,
                      label=MODEL_NAMES.get(model, model))
            axis.fill_between(display_periods,
                              np.maximum(mean - 1.96 * se, np.finfo(float).tiny),
                              mean + 1.96 * se,
                              color=MODEL_COLORS.get(model), alpha=0.12, linewidth=0)
        axis.axhline(1.0, color="#777777", linewidth=0.7, linestyle="--", zorder=0)
        count_text = "/".join(f"{value:,}" for value in counts)
        axis.set_title(
            f"{LANGUAGE_NAMES.get(language, language)}\n$n_{{L}}/n_{{Q}}={count_text}$",
            fontsize=9.5,
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(True, which="major", linewidth=0.35, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Period (model tokens)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Relative spectral density")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Document-weighted mean surprisal spectra across Gutenberg languages", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(
    output: Path,
    exact: dict[str, tuple[np.ndarray, np.ndarray, int]],
    binned: dict[str, tuple[np.ndarray, np.ndarray, int]],
    exact_periods: np.ndarray,
    binned_periods: np.ndarray,
    languages: list[str],
) -> None:
    import matplotlib.pyplot as plt

    exact_display = (exact_periods > 2.0) & (exact_periods <= 2048.0)
    binned_display = (binned_periods > 2.0) & (binned_periods <= 2048.0)
    shown_exact_periods = exact_periods[exact_display]
    shown_binned_periods = binned_periods[binned_display]
    selected = [language for language in ("en", "zh") if language in exact]
    fig, axes = plt.subplots(1, len(selected), figsize=(10.0, 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, language in zip(axes, selected):
        raw_mean, _, documents = exact[language]
        mean, _, _ = binned[language]
        raw_mean = raw_mean[exact_display]
        mean = mean[binned_display]
        axis.plot(shown_exact_periods, raw_mean, color="#999999", alpha=0.45,
                  linewidth=0.55, label="Exact-bin mean")
        axis.plot(shown_binned_periods, mean, color="#000000", linewidth=1.2,
                  marker="o", markersize=2.0, label="Log-period bins")

        window = min(15, len(mean) if len(mean) % 2 == 1 else len(mean) - 1)
        if window >= 5:
            smoothed = np.exp(savgol_filter(np.log(mean), window_length=window, polyorder=3))
            axis.plot(shown_binned_periods, smoothed, color="#009E73", linewidth=1.4,
                      label="Savitzky--Golay")

        x = np.log2(shown_binned_periods)
        spline = UnivariateSpline(x, np.log(mean), s=len(x) * 0.02)
        axis.plot(shown_binned_periods, np.exp(spline(x)), color="#CC79A7", linewidth=1.4,
                  linestyle="--", label="Smoothing spline")
        axis.axhline(1.0, color="#777777", linewidth=0.7, linestyle=":")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"{LANGUAGE_NAMES.get(language, language)} ($n={documents:,}$)")
        axis.set_xlabel("Period (model tokens)")
        axis.grid(True, which="major", linewidth=0.35, alpha=0.3)
    axes[0].set_ylabel("Relative spectral density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Comparison of spectrum aggregation and smoothing methods", y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = list(dict.fromkeys(args.languages))
    if len(languages) != 10:
        raise ValueError("The publication facet layout currently requires exactly 10 languages")

    if args.plot_only:
        exact_summaries, binned_summaries, model_periods, binned_periods = read_plot_summaries(
            args.output_dir, args.models, languages
        )
    else:
        exact_rows: list[dict[str, object]] = []
        binned_rows: list[dict[str, object]] = []
        exact_summaries = {}
        binned_summaries = {}
        model_periods = {}
        binned_periods = None
        bin_counts = None

        for model in args.models:
            exact, binned, periods, centers, counts = aggregate_model(
                args.input_root, model, set(languages), args.log_bins, args.fft_batch_size
            )
            model_periods[model] = periods
            binned_periods = centers
            bin_counts = counts
            exact_summaries[model] = {}
            binned_summaries[model] = {}
            for language in languages:
                if language not in exact:
                    continue
                mean, se = exact[language].mean_and_se()
                exact_summaries[model][language] = (mean, se, exact[language].count)
                for k, period, value, error in zip(range(1, len(periods) + 1), periods, mean, se):
                    exact_rows.append({
                        "model": model, "language": language, "documents": exact[language].count,
                        "fourier_k": k, "frequency": k / (2 * len(periods)), "period": period,
                        "mean_relative_density": value, "se_relative_density": error,
                    })

                mean, se = binned[language].mean_and_se()
                binned_summaries[model][language] = (mean, se, binned[language].count)
                for index, (period, bins, value, error) in enumerate(zip(centers, counts, mean, se)):
                    binned_rows.append({
                        "model": model, "language": language, "documents": binned[language].count,
                        "log_bin": index, "period_center": period, "fourier_bins": int(bins),
                        "mean_relative_density": value, "se_relative_density": error,
                    })

        write_rows(args.output_dir / "language_spectra_exact.csv", exact_rows)
        write_rows(args.output_dir / "language_spectra_logbin.csv", binned_rows)
        first_model = args.models[0]
        summary = {
            "models": args.models,
            "languages": languages,
            "document_counts": {
                model: {language: values[2] for language, values in by_language.items()}
                for model, by_language in binned_summaries.items()
            },
            "window_length": int(2 * len(model_periods[first_model])),
            "exact_fourier_bins": int(len(model_periods[first_model])),
            "requested_log_period_bins": args.log_bins,
            "populated_log_period_bins": int(len(binned_periods)),
            "relative_density_reference": 1.0,
            "aggregation": "three windows averaged per document, then documents averaged per language",
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "language_spectra_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))

    if not args.aggregate_only:
        plot_language_facets(
            args.output_dir / "gutenberg_language_spectra",
            binned_summaries,
            binned_periods,
            languages,
        )
        first_model = args.models[0]
        plot_method_comparison(
            args.output_dir / "gutenberg_spectrum_method_comparison",
            exact_summaries[first_model],
            binned_summaries[first_model],
            model_periods[first_model],
            binned_periods,
            languages,
        )


if __name__ == "__main__":
    main()
