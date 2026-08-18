#!/usr/bin/env python3
"""Prepare fixed Gutenberg windows for causal-LM surprisal scoring.

Window centers are anchored to raw-text positions so different tokenizers start
from comparable parts of a document.  The saved target is always preceded by a
fixed context that is scored by the model but excluded from spectral analysis.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


POSITIONS = (("front", 1.0 / 6.0), ("middle", 1.0 / 2.0), ("rear", 5.0 / 6.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--target-length", type=int, default=4096)
    parser.add_argument("--allow-skips", action="store_true")
    return parser.parse_args()


def nearest_paragraph_boundary(text: str, fraction: float, radius: int = 8192) -> int:
    target = int(round(len(text) * fraction))
    low = max(0, target - radius)
    high = min(len(text), target + radius)
    left = text.rfind("\n\n", low, target)
    right = text.find("\n\n", target, high)
    candidates = []
    if left >= 0:
        candidates.append(left + 2)
    if right >= 0:
        candidates.append(right + 2)
    return min(candidates, key=lambda value: abs(value - target)) if candidates else target


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"document_id", "language"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}: {path}")
    return rows


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ checkpoint without exposing a partially written file."""
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.context_length < 1 or args.target_length < 2:
        raise ValueError("context-length must be >=1 and target-length must be >=2")

    started = time.perf_counter()
    rows = load_manifest(args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for character offset alignment")
    tokenizer.model_max_length = 10**12

    sequences: list[np.ndarray] = []
    document_ids: list[int] = []
    languages: list[str] = []
    position_names: list[str] = []
    anchor_chars: list[int] = []
    token_centers: list[int] = []
    target_starts: list[int] = []
    document_token_counts: list[int] = []
    errors: list[dict[str, str]] = []
    document_details: list[dict[str, object]] = []

    tokenization_seconds = 0.0
    reading_seconds = 0.0
    expected_sequence_length = args.context_length + args.target_length

    for row_number, row in enumerate(rows, start=1):
        document_id = int(row["document_id"])
        path = args.corpus_dir / f"{document_id}.txt"
        try:
            read_started = time.perf_counter()
            text = path.read_text(encoding="utf-8", errors="replace")
            reading_seconds += time.perf_counter() - read_started

            tokenize_started = time.perf_counter()
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            tokenization_seconds += time.perf_counter() - tokenize_started
            ids = np.asarray(encoded["input_ids"], dtype=np.int32)
            offsets = encoded["offset_mapping"]
            starts = [int(pair[0]) for pair in offsets]
            required = args.context_length + args.target_length
            if len(ids) < required:
                raise ValueError(f"only {len(ids)} tokens; need at least {required}")

            # Stage every array for this document locally.  Nothing enters the
            # output until all three windows and the non-overlap check pass.
            # This makes --allow-skips transactional at document granularity.
            per_document_sequences: list[np.ndarray] = []
            per_document_position_names: list[str] = []
            per_document_anchor_chars: list[int] = []
            per_document_token_centers: list[int] = []
            per_document_starts: list[int] = []
            detail = {
                "document_id": document_id,
                "language": row["language"],
                "characters": len(text),
                "tokens": int(len(ids)),
                "windows": [],
            }
            for position_name, fraction in POSITIONS:
                anchor = nearest_paragraph_boundary(text, fraction)
                center = bisect.bisect_left(starts, anchor)
                center = min(max(center, 0), len(ids) - 1)
                target_start = center - args.target_length // 2
                target_start = max(args.context_length, target_start)
                target_start = min(target_start, len(ids) - args.target_length)
                input_start = target_start - args.context_length
                input_end = target_start + args.target_length
                sequence = ids[input_start:input_end]
                if len(sequence) != expected_sequence_length:
                    raise AssertionError(
                        f"window {position_name} has {len(sequence)} tokens, "
                        f"expected {expected_sequence_length}"
                    )
                per_document_sequences.append(sequence)
                per_document_position_names.append(position_name)
                per_document_anchor_chars.append(anchor)
                per_document_token_centers.append(center)
                per_document_starts.append(target_start)
                detail["windows"].append(
                    {
                        "position": position_name,
                        "anchor_char": anchor,
                        "token_center": center,
                        "target_start": target_start,
                    }
                )

            if any(
                later < earlier + args.target_length
                for earlier, later in zip(per_document_starts, per_document_starts[1:])
            ):
                raise ValueError(f"target windows overlap: {per_document_starts}")

            sequences.extend(per_document_sequences)
            document_ids.extend([document_id] * len(POSITIONS))
            languages.extend([row["language"]] * len(POSITIONS))
            position_names.extend(per_document_position_names)
            anchor_chars.extend(per_document_anchor_chars)
            token_centers.extend(per_document_token_centers)
            target_starts.extend(per_document_starts)
            document_token_counts.extend([len(ids)] * len(POSITIONS))
            document_details.append(detail)
        except Exception as exc:  # preserve the full benchmark manifest for diagnosis
            errors.append(
                {
                    "row": str(row_number),
                    "document_id": str(document_id),
                    "language": row.get("language", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if errors and not args.allow_skips:
        preview = "\n".join(str(item) for item in errors[:10])
        raise RuntimeError(f"{len(errors)} documents failed preparation:\n{preview}")
    if not sequences:
        raise RuntimeError("No windows were prepared")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_started = time.perf_counter()
    atomic_savez(
        args.output,
        input_ids=np.stack(sequences),
        document_id=np.asarray(document_ids, dtype=np.int32),
        language=np.asarray(languages),
        position=np.asarray(position_names),
        anchor_char=np.asarray(anchor_chars, dtype=np.int64),
        token_center=np.asarray(token_centers, dtype=np.int64),
        target_start=np.asarray(target_starts, dtype=np.int64),
        document_token_count=np.asarray(document_token_counts, dtype=np.int64),
        context_length=np.asarray(args.context_length, dtype=np.int32),
        target_length=np.asarray(args.target_length, dtype=np.int32),
    )
    save_seconds = time.perf_counter() - save_started
    summary = {
        "host": socket.gethostname(),
        "manifest": str(args.manifest),
        "corpus_dir": str(args.corpus_dir),
        "tokenizer": str(args.tokenizer),
        "tokenizer_class": type(tokenizer).__name__,
        "output": str(args.output),
        "requested_documents": len(rows),
        "prepared_documents": len(document_details),
        "prepared_windows": len(sequences),
        "context_length": args.context_length,
        "target_length": args.target_length,
        "reading_seconds": reading_seconds,
        "tokenization_seconds": tokenization_seconds,
        "save_seconds": save_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "errors": errors,
        "documents": document_details,
    }
    summary_path = args.output.with_suffix(".summary.json")
    atomic_write_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in (
        "requested_documents", "prepared_documents", "prepared_windows",
        "reading_seconds", "tokenization_seconds", "save_seconds", "elapsed_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
