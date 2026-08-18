#!/usr/bin/env python3
"""Run the ten-band shuffle-spectrum diagnostic on benchmark surprisal shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp


PERIOD_BANDS = (
    ("1024--2048", 1024.0, 2048.0),
    ("512--1024", 512.0, 1024.0),
    ("256--512", 256.0, 512.0),
    ("128--256", 128.0, 256.0),
    ("64--128", 64.0, 128.0),
    ("32--64", 32.0, 64.0),
    ("16--32", 16.0, 32.0),
    ("8--16", 8.0, 16.0),
    ("4--8", 4.0, 8.0),
    ("2--4", 2.0, 4.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def normalized_power(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    std = values.std(axis=-1, keepdims=True)
    if np.any(std == 0):
        raise ValueError("A surprisal sequence has zero variance")
    standardized = (values - values.mean(axis=-1, keepdims=True)) / std
    power = np.abs(np.fft.rfft(standardized, axis=-1)) ** 2
    power[..., 0] = 0.0
    power /= np.maximum(power.sum(axis=-1, keepdims=True), np.finfo(float).tiny)
    frequency = np.fft.rfftfreq(values.shape[-1], d=1.0)
    return frequency, power


def band_masks(frequency: np.ndarray) -> list[np.ndarray]:
    masks = []
    for _, period_low, period_high in PERIOD_BANDS:
        masks.append((frequency >= 1.0 / period_high) & (frequency < 1.0 / period_low))
    return masks


def band_mass(power: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    return np.stack([power[..., mask].sum(axis=-1) for mask in masks], axis=-1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    shard_paths = sorted(args.input_dir.glob(f"{args.model_label}.rank*-of-*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No score shards in {args.input_dir}")

    arrays = defaultdict(list)
    for path in shard_paths:
        shard = np.load(path, allow_pickle=False)
        for key in ("nll", "document_id", "language", "position", "document_token_count"):
            arrays[key].append(shard[key])
    nll = np.concatenate(arrays["nll"])
    doc_ids = np.concatenate(arrays["document_id"])
    languages = np.concatenate(arrays["language"])
    positions = np.concatenate(arrays["position"])
    doc_lengths = np.concatenate(arrays["document_token_count"])
    order = np.lexsort((positions, doc_ids))
    nll, doc_ids, languages, positions, doc_lengths = (
        value[order] for value in (nll, doc_ids, languages, positions, doc_lengths)
    )

    frequency, original_power = normalized_power(nll[:1])
    masks = band_masks(frequency)
    bin_counts = [int(mask.sum()) for mask in masks]
    expected_counts = [2 ** (index + 1) for index in range(len(PERIOD_BANDS))]
    if bin_counts != expected_counts:
        raise AssertionError(f"Unexpected Fourier-bin counts: {bin_counts}")

    window_rows = []
    document_values: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for index, values in enumerate(nll):
        frequency, power = normalized_power(values[None, :])
        original_mass = band_mass(power, masks)[0]
        position_index = {"front": 0, "middle": 1, "rear": 2}[str(positions[index])]
        rng = np.random.default_rng(args.seed + int(doc_ids[index]) * 10 + position_index)
        shuffled = np.empty((args.shuffles, len(values)), dtype=np.float64)
        for shuffle_index in range(args.shuffles):
            shuffled[shuffle_index] = rng.permutation(values)
        _, shuffled_power = normalized_power(shuffled)
        shuffled_mass = band_mass(shuffled_power, masks)
        delta = np.log(original_mass) - np.log(shuffled_mass).mean(axis=0)
        document_values[(int(doc_ids[index]), str(languages[index]))].append(delta)
        row = {
            "model": args.model_label,
            "document_id": int(doc_ids[index]),
            "language": str(languages[index]),
            "position": str(positions[index]),
            "document_tokens": int(doc_lengths[index]),
        }
        for (label, _, _), value in zip(PERIOD_BANDS, delta):
            row[f"delta_{label}"] = float(value)
        window_rows.append(row)

    document_rows = []
    effects_by_language: dict[str, list[np.ndarray]] = defaultdict(list)
    for (document_id, language), values in sorted(document_values.items()):
        mean_delta = np.mean(values, axis=0)
        effects_by_language[language].append(mean_delta)
        row = {
            "model": args.model_label,
            "document_id": document_id,
            "language": language,
            "windows": len(values),
        }
        for (label, _, _), value in zip(PERIOD_BANDS, mean_delta):
            row[f"mean_delta_{label}"] = float(value)
        document_rows.append(row)

    test_rows = []
    for language, values in sorted(effects_by_language.items()):
        matrix = np.asarray(values)
        for band_index, (label, low, high) in enumerate(PERIOD_BANDS):
            result = ttest_1samp(matrix[:, band_index], popmean=0.0)
            test_rows.append(
                {
                    "model": args.model_label,
                    "language": language,
                    "period_band": label,
                    "period_low_open": low,
                    "period_high_closed": high,
                    "fourier_bins": bin_counts[band_index],
                    "documents": len(matrix),
                    "mean_delta": float(matrix[:, band_index].mean()),
                    "geometric_power_ratio": float(math.exp(matrix[:, band_index].mean())),
                    "t_statistic": float(result.statistic),
                    "p_raw": float(result.pvalue),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "window_deltas.csv", window_rows)
    write_csv(args.output_dir / "document_deltas.csv", document_rows)
    write_csv(args.output_dir / "language_band_diagnostics.csv", test_rows)
    summary = {
        "model": args.model_label,
        "input_dir": str(args.input_dir),
        "shards": [str(path) for path in shard_paths],
        "documents": len(document_values),
        "windows": len(nll),
        "languages": sorted(effects_by_language),
        "window_length": int(nll.shape[1]),
        "shuffles": args.shuffles,
        "seed": args.seed,
        "period_bands": [
            {
                "label": label,
                "period": f"({low:g}, {high:g}]",
                "frequency": f"[1/{high:g}, 1/{low:g})",
                "fourier_bins": bins,
            }
            for (label, low, high), bins in zip(PERIOD_BANDS, bin_counts)
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "warning": "Benchmark diagnostics only; not a confirmatory corpus result.",
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
