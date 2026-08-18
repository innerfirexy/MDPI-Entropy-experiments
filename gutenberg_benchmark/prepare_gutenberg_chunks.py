#!/usr/bin/env python3
"""Prepare a full Gutenberg manifest as resumable tokenizer-specific chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


POSITIONS = {"front", "middle", "rear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--target-length", type=int, default=4096)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail a chunk instead of recording and skipping ineligible documents.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows or not {"document_id", "language"}.issubset(fieldnames):
        raise ValueError("manifest must contain document_id and language")
    document_ids = [row["document_id"] for row in rows]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("manifest contains duplicate document_id values")
    return fieldnames, rows


def validate_prepared(
    output: Path,
    summary_path: Path,
    expected_rows: list[dict[str, str]],
    context_length: int,
    target_length: int,
) -> tuple[dict[str, object], set[int]]:
    if not output.is_file() or not summary_path.is_file():
        raise RuntimeError("prepared checkpoint files are absent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(output, allow_pickle=False) as prepared:
        required = {
            "input_ids", "document_id", "language", "position", "anchor_char",
            "token_center", "target_start", "document_token_count",
            "context_length", "target_length",
        }
        if not required.issubset(prepared.files):
            raise RuntimeError(f"prepared checkpoint is missing {sorted(required - set(prepared.files))}")
        windows = len(prepared["document_id"])
        if prepared["input_ids"].shape != (windows, context_length + target_length):
            raise RuntimeError("prepared input_ids shape is invalid")
        if int(prepared["context_length"]) != context_length:
            raise RuntimeError("prepared context length mismatch")
        if int(prepared["target_length"]) != target_length:
            raise RuntimeError("prepared target length mismatch")
        for key in required - {"input_ids", "context_length", "target_length"}:
            if len(prepared[key]) != windows:
                raise RuntimeError(f"prepared metadata length mismatch: {key}")

        document_ids = [int(value) for value in prepared["document_id"]]
        eligible_ids = set(document_ids)
        expected_ids = {int(row["document_id"]) for row in expected_rows}
        if not eligible_ids.issubset(expected_ids):
            raise RuntimeError("prepared checkpoint contains a document outside its chunk manifest")
        counts = Counter(document_ids)
        positions: dict[int, set[str]] = defaultdict(set)
        for document_id, position in zip(document_ids, prepared["position"]):
            positions[document_id].add(str(position))
        for document_id in eligible_ids:
            if counts[document_id] != 3:
                raise RuntimeError(
                    f"document {document_id} has {counts[document_id]} windows instead of 3"
                )
            if positions[document_id] != POSITIONS:
                raise RuntimeError(f"document {document_id} does not have front/middle/rear windows")

    detailed_ids = {int(item["document_id"]) for item in summary.get("documents", [])}
    if detailed_ids != eligible_ids:
        raise RuntimeError("summary document details do not match prepared arrays")
    if summary.get("requested_documents") != len(expected_rows):
        raise RuntimeError("summary requested-document count mismatch")
    if summary.get("prepared_documents") != len(eligible_ids):
        raise RuntimeError("summary prepared-document count mismatch")
    if summary.get("prepared_windows") != len(eligible_ids) * 3:
        raise RuntimeError("summary prepared-window count mismatch")
    error_ids = {int(item["document_id"]) for item in summary.get("errors", [])}
    if eligible_ids & error_ids:
        raise RuntimeError("a skipped document leaked into the prepared arrays")
    if eligible_ids | error_ids != {int(row["document_id"]) for row in expected_rows}:
        raise RuntimeError("summary does not account for every requested document")
    return summary, eligible_ids


def main() -> None:
    args = parse_args()
    if args.chunk_size < 8:
        raise ValueError("chunk-size must be at least 8 for the 8-GPU scorer")
    fieldnames, rows = load_manifest(args.manifest)
    manifest_dir = args.output_root / "manifests" / args.tokenizer_label
    prepared_dir = args.output_root / "prepared" / args.tokenizer_label
    manifest_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    preparer = Path(__file__).with_name("prepare_gutenberg_windows.py")

    eligible_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    chunk_results = []
    total_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size

    for chunk_index in range(total_chunks):
        chunk_name = f"chunk-{chunk_index:05d}"
        chunk_rows = rows[chunk_index * args.chunk_size : (chunk_index + 1) * args.chunk_size]
        chunk_manifest = manifest_dir / f"{chunk_name}.csv"
        output = prepared_dir / f"{chunk_name}.npz"
        summary_path = output.with_suffix(".summary.json")
        success_path = prepared_dir / f"{chunk_name}._SUCCESS.json"
        atomic_write_csv(chunk_manifest, fieldnames, chunk_rows)
        manifest_hash = sha256(chunk_manifest)

        complete = False
        if not args.force and success_path.is_file():
            try:
                marker = json.loads(success_path.read_text(encoding="utf-8"))
                if all((
                    marker.get("manifest_sha256") == manifest_hash,
                    marker.get("tokenizer") == str(args.tokenizer),
                    marker.get("context_length") == args.context_length,
                    marker.get("target_length") == args.target_length,
                )):
                    summary, eligible_ids = validate_prepared(
                        output, summary_path, chunk_rows, args.context_length, args.target_length
                    )
                    complete = True
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
                complete = False

        if not complete:
            command = [
                sys.executable,
                str(preparer),
                "--manifest", str(chunk_manifest),
                "--corpus-dir", str(args.corpus_dir),
                "--tokenizer", str(args.tokenizer),
                "--output", str(output),
                "--context-length", str(args.context_length),
                "--target-length", str(args.target_length),
            ]
            if not args.strict:
                command.append("--allow-skips")
            subprocess.run(command, check=True)
            summary, eligible_ids = validate_prepared(
                output, summary_path, chunk_rows, args.context_length, args.target_length
            )
            marker = {
                "status": "complete",
                "created_at_unix": time.time(),
                "chunk": chunk_name,
                "manifest": str(chunk_manifest),
                "manifest_sha256": manifest_hash,
                "tokenizer": str(args.tokenizer),
                "tokenizer_label": args.tokenizer_label,
                "context_length": args.context_length,
                "target_length": args.target_length,
                "requested_documents": len(chunk_rows),
                "prepared_documents": len(eligible_ids),
                "skipped_documents": len(summary.get("errors", [])),
                "prepared_windows": len(eligible_ids) * 3,
                "output": str(output),
                "summary": str(summary_path),
            }
            atomic_write_json(success_path, marker)

        row_by_id = {int(row["document_id"]): row for row in chunk_rows}
        for document_id in sorted(eligible_ids):
            eligible_rows.append({**row_by_id[document_id], "chunk": chunk_name})
        for error in summary.get("errors", []):
            document_id = int(error["document_id"])
            skipped_rows.append({
                **row_by_id[document_id],
                "chunk": chunk_name,
                "error": error["error"],
            })
        chunk_results.append(json.loads(success_path.read_text(encoding="utf-8")))
        print(json.dumps({
            "chunk": chunk_name,
            "status": "resumed" if complete else "prepared",
            "prepared_documents": len(eligible_ids),
            "skipped_documents": len(summary.get("errors", [])),
        }))

    eligible_fields = fieldnames + ["chunk"]
    skipped_fields = fieldnames + ["chunk", "error"]
    atomic_write_csv(
        args.output_root / "manifests" / f"{args.tokenizer_label}_eligible.csv",
        eligible_fields,
        eligible_rows,
    )
    atomic_write_csv(
        args.output_root / "manifests" / f"{args.tokenizer_label}_skipped.csv",
        skipped_fields,
        skipped_rows,
    )
    index = {
        "status": "complete",
        "manifest": str(args.manifest),
        "tokenizer": str(args.tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "chunk_size": args.chunk_size,
        "chunks": len(chunk_results),
        "requested_documents": len(rows),
        "prepared_documents": len(eligible_rows),
        "skipped_documents": len(skipped_rows),
        "prepared_windows": len(eligible_rows) * 3,
        "chunk_checkpoints": chunk_results,
    }
    atomic_write_json(prepared_dir / "index.json", index)
    print(json.dumps({key: index[key] for key in (
        "chunks", "requested_documents", "prepared_documents",
        "skipped_documents", "prepared_windows",
    )}, indent=2))


if __name__ == "__main__":
    main()
