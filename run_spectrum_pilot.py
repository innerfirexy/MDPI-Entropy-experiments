#!/usr/bin/env python3
"""Document-level spectrum validation against within-document shuffles.

The pilot uses existing Llama-3-8B-base token surprisal sequences for PTB-Brown
and PTB-WSJ.  Each eligible document contributes one fixed-length sequence,
which is z-normalized before its normalized one-sided periodogram is computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon


OCTAVE_BANDS = (
    ("2--4", 2.0, 4.0),
    ("4--8", 4.0, 8.0),
    ("8--16", 8.0, 16.0),
    ("16--32", 16.0, 32.0),
    ("32--64", 32.0, 64.0),
    ("64--128", 64.0, 128.0),
    ("128--256", 128.0, 256.0),
    ("256--512", 256.0, 512.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/xy/projects/period-playground/cl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot_brown_wsj"))
    parser.add_argument("--window-length", type=int, default=1024)
    parser.add_argument("--shuffles", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def normalized_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return non-DC frequencies and PSD normalized to unit mass."""
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean()) / x.std(ddof=0)
    power = np.abs(np.fft.rfft(x)) ** 2
    power[..., 0] = 0.0
    denom = power.sum(axis=-1, keepdims=True)
    power = power / np.maximum(denom, np.finfo(float).tiny)
    freq = np.fft.rfftfreq(x.shape[-1], d=1.0)
    return freq[1:], power[..., 1:]


def band_masks(freq: np.ndarray) -> list[np.ndarray]:
    periods = 1.0 / freq
    return [(periods >= low) & (periods < high) for _, low, high in OCTAVE_BANDS]


def band_mass(power: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    return np.stack([power[..., mask].sum(axis=-1) for mask in masks], axis=-1)


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - half, center + half


def analyze_document(
    values: np.ndarray,
    masks: list[np.ndarray],
    rng: np.random.Generator,
    shuffles: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return original masses, log mass deltas, and an omnibus empirical p-value."""
    freq, original_power = normalized_psd(values)
    original_mass = band_mass(original_power, masks)

    surrogate_values = np.empty((shuffles, len(values)), dtype=np.float64)
    for i in range(shuffles):
        surrogate_values[i] = rng.permutation(values)
    _, surrogate_power = normalized_psd(surrogate_values)
    surrogate_mass = band_mass(surrogate_power, masks)

    eps = np.finfo(float).tiny
    log_original = np.log(original_mass + eps)
    log_surrogate = np.log(surrogate_mass + eps)
    delta = log_original - log_surrogate.mean(axis=0)

    # Omnibus statistic across correlated octave bands.  Each null statistic
    # uses leave-one-out moments so a surrogate is not standardized by itself.
    null_sum = log_surrogate.sum(axis=0)
    null_sumsq = np.square(log_surrogate).sum(axis=0)
    null_mean = log_surrogate.mean(axis=0)
    null_sd = log_surrogate.std(axis=0, ddof=1)
    observed_t = np.square((log_original - null_mean) / np.maximum(null_sd, 1e-12)).sum()

    loo_mean = (null_sum - log_surrogate) / (shuffles - 1)
    loo_ss = (null_sumsq - np.square(log_surrogate)) - (shuffles - 1) * np.square(loo_mean)
    loo_sd = np.sqrt(np.maximum(loo_ss / (shuffles - 2), 1e-24))
    null_t = np.square((log_surrogate - loo_mean) / loo_sd).sum(axis=1)
    omnibus_p = (1.0 + np.count_nonzero(null_t >= observed_t)) / (shuffles + 1.0)
    return original_mass, delta, omnibus_p


def load_corpus(path: Path, window_length: int) -> tuple[list[tuple[str, np.ndarray]], dict]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    lengths = np.asarray([len(v) for v in raw.values()])
    documents = []
    for doc_id in sorted(raw, key=str):
        values = np.asarray(raw[doc_id], dtype=np.float64)
        if len(values) < window_length or not np.all(np.isfinite(values)) or values.std() == 0:
            continue
        documents.append((str(doc_id), values[:window_length]))
    metadata = {
        "source_documents": int(len(raw)),
        "eligible_documents": int(len(documents)),
        "source_tokens": int(lengths.sum()),
        "minimum_length": int(lengths.min()),
        "median_length": float(np.median(lengths)),
        "maximum_length": int(lengths.max()),
    }
    return documents, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(test_rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    corpora = ("brown", "wsj")
    colors = {"brown": "#2b6cb0", "wsj": "#c05621"}
    x = np.arange(len(OCTAVE_BANDS))
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for offset, corpus in zip((-0.08, 0.08), corpora):
        rows = [r for r in test_rows if r["corpus"] == corpus]
        means = np.asarray([r["mean_log_ratio"] for r in rows], dtype=float)
        lows = np.asarray([r["ci_low"] for r in rows], dtype=float)
        highs = np.asarray([r["ci_high"] for r in rows], dtype=float)
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=colors[corpus],
            label=corpus.upper(),
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, [label for label, _, _ in OCTAVE_BANDS], rotation=30)
    ax.set_xlabel("Period band (tokens)")
    ax.set_ylabel("Mean log power ratio: original / shuffled")
    ax.set_title("Document-level spectrum validation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.shuffles < 3:
        raise ValueError("At least three shuffles are needed for leave-one-out null statistics.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    probe_freq = np.fft.rfftfreq(args.window_length, d=1.0)[1:]
    masks = band_masks(probe_freq)
    if any(not np.any(mask) for mask in masks):
        raise ValueError("Window length is too short for the requested octave bands.")

    document_rows: list[dict] = []
    test_rows: list[dict] = []
    corpus_metadata = {}

    for corpus_index, corpus in enumerate(("brown", "wsj")):
        source = args.data_dir / f"{corpus}_nlls_doc.pkl"
        documents, metadata = load_corpus(source, args.window_length)
        corpus_metadata[corpus] = metadata
        corpus_rng = np.random.default_rng(args.seed + corpus_index * 1_000_000)
        corpus_deltas = []
        omnibus_ps = []

        for doc_index, (doc_id, values) in enumerate(documents):
            doc_seed = int(corpus_rng.integers(0, np.iinfo(np.int64).max))
            doc_rng = np.random.default_rng(doc_seed)
            masses, deltas, omnibus_p = analyze_document(values, masks, doc_rng, args.shuffles)
            corpus_deltas.append(deltas)
            omnibus_ps.append(omnibus_p)
            row = {
                "corpus": corpus,
                "document_id": doc_id,
                "window_length": args.window_length,
                "omnibus_p": omnibus_p,
            }
            for (label, _, _), mass, delta in zip(OCTAVE_BANDS, masses, deltas):
                safe_label = label.replace("--", "_")
                row[f"mass_{safe_label}"] = mass
                row[f"log_ratio_{safe_label}"] = delta
            document_rows.append(row)

        deltas = np.asarray(corpus_deltas)
        bootstrap_rng = np.random.default_rng(args.seed + 10_000 + corpus_index)
        for band_index, (label, low, high) in enumerate(OCTAVE_BANDS):
            values = deltas[:, band_index]
            t_result = ttest_1samp(values, popmean=0.0, alternative="two-sided")
            try:
                w_result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
                wilcoxon_p = float(w_result.pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            ci_low, ci_high = bootstrap_mean_ci(values, bootstrap_rng, args.bootstrap)
            test_rows.append(
                {
                    "corpus": corpus,
                    "period_band": label,
                    "period_low": low,
                    "period_high": high,
                    "n_documents": len(values),
                    "mean_log_ratio": float(values.mean()),
                    "geometric_power_ratio": float(np.exp(values.mean())),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "cohen_dz": float(values.mean() / values.std(ddof=1)),
                    "t_statistic": float(t_result.statistic),
                    "t_p_raw": float(t_result.pvalue),
                    "wilcoxon_p_raw": wilcoxon_p,
                }
            )

        omnibus_ps = np.asarray(omnibus_ps)
        successes = int(np.count_nonzero(omnibus_ps < 0.05))
        prevalence_low, prevalence_high = wilson_ci(successes, len(omnibus_ps))
        metadata.update(
            {
                "omnibus_significant_documents": successes,
                "omnibus_prevalence": successes / len(omnibus_ps),
                "omnibus_prevalence_ci_low": prevalence_low,
                "omnibus_prevalence_ci_high": prevalence_high,
            }
        )

    t_adjusted = bh_adjust([row["t_p_raw"] for row in test_rows])
    w_adjusted = bh_adjust([row["wilcoxon_p_raw"] for row in test_rows])
    for row, t_fdr, w_fdr in zip(test_rows, t_adjusted, w_adjusted):
        row["t_p_fdr"] = float(t_fdr)
        row["wilcoxon_p_fdr"] = float(w_fdr)

    write_csv(args.output_dir / "document_results.csv", document_rows)
    write_csv(args.output_dir / "band_tests.csv", test_rows)
    make_plot(test_rows, figure_dir / "band_log_power_ratios.pdf")

    config = {
        "data_dir": str(args.data_dir),
        "window_length": args.window_length,
        "window_policy": "first eligible tokens",
        "shuffles_per_document": args.shuffles,
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "preprocessing": "within-document z-score",
        "spectrum": "normalized one-sided periodogram; DC excluded",
        "bands": [
            {"label": label, "period_low": low, "period_high": high}
            for label, low, high in OCTAVE_BANDS
        ],
        "multiple_testing": "Benjamini-Hochberg across both corpora and all octave bands",
        "corpora": corpus_metadata,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    report_lines = ["# Brown/WSJ spectrum-validation pilot", ""]
    for corpus in ("brown", "wsj"):
        meta = corpus_metadata[corpus]
        report_lines.extend(
            [
                f"## {corpus.upper()}",
                "",
                f"- Eligible documents: {meta['eligible_documents']} / {meta['source_documents']}",
                f"- Omnibus-significant documents: {meta['omnibus_significant_documents']} "
                f"({meta['omnibus_prevalence']:.1%}, 95% Wilson CI "
                f"{meta['omnibus_prevalence_ci_low']:.1%}--{meta['omnibus_prevalence_ci_high']:.1%})",
                "",
                "| Period (tokens) | Mean log ratio | Power ratio | 95% CI | dz | FDR p |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in (r for r in test_rows if r["corpus"] == corpus):
            report_lines.append(
                f"| {row['period_band']} | {row['mean_log_ratio']:.4f} | "
                f"{row['geometric_power_ratio']:.3f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                f"{row['cohen_dz']:.3f} | {row['t_p_fdr']:.3g} |"
            )
        report_lines.append("")
    (args.output_dir / "pilot_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
