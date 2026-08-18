#!/usr/bin/env python3
"""Validate all rank shards for one prepared chunk and seal the checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


METADATA_KEYS = (
    "document_id",
    "language",
    "position",
    "anchor_char",
    "token_center",
    "target_start",
    "document_token_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--write-success", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepared_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main() -> None:
    args = parse_args()
    success_path = args.output_dir / "_SUCCESS.json"
    fingerprint = prepared_fingerprint(args.prepared)
    if args.require_success:
        if not success_path.is_file():
            raise RuntimeError(f"success marker is absent: {success_path}")
        marker = json.loads(success_path.read_text(encoding="utf-8"))
        if marker.get("prepared") != fingerprint:
            raise RuntimeError("success marker belongs to a different prepared checkpoint")
        if marker.get("model") != str(args.model) or marker.get("model_label") != args.model_label:
            raise RuntimeError("success marker belongs to a different model")
        if marker.get("world_size") != args.world_size:
            raise RuntimeError("success marker has a different world size")

    with np.load(args.prepared, allow_pickle=False) as prepared:
        all_doc_ids = prepared["document_id"]
        ordered_docs = list(dict.fromkeys(int(value) for value in all_doc_ids))
        target_length = int(prepared["target_length"])
        total_windows = 0
        rank_details = []

        for rank in range(args.world_size):
            assigned_docs = set(ordered_docs[rank :: args.world_size])
            indices = np.asarray(
                [i for i, value in enumerate(all_doc_ids) if int(value) in assigned_docs],
                dtype=np.int64,
            )
            if not len(indices):
                raise RuntimeError(f"rank {rank} has no assigned windows")
            stem = f"{args.model_label}.rank{rank:02d}-of-{args.world_size:02d}"
            output_path = args.output_dir / f"{stem}.npz"
            summary_path = args.output_dir / f"{stem}.summary.json"
            if not output_path.is_file() or not summary_path.is_file():
                raise RuntimeError(f"rank {rank} checkpoint is incomplete")

            with np.load(output_path, allow_pickle=False) as output:
                if output["nll"].shape != (len(indices), target_length):
                    raise RuntimeError(f"rank {rank} has an invalid nll shape")
                if not np.isfinite(output["nll"]).all():
                    raise RuntimeError(f"rank {rank} contains non-finite nll values")
                for key in METADATA_KEYS:
                    if not np.array_equal(output[key], prepared[key][indices]):
                        raise RuntimeError(f"rank {rank} metadata mismatch: {key}")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "model": str(args.model),
                "model_label": args.model_label,
                "prepared": str(args.prepared),
                "rank": rank,
                "world_size": args.world_size,
                "documents": len(assigned_docs),
                "windows": len(indices),
                "target_length": target_length,
            }
            mismatches = {
                key: (summary.get(key), value)
                for key, value in expected.items()
                if summary.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"rank {rank} summary mismatch: {mismatches}")
            total_windows += len(indices)
            rank_details.append({
                "rank": rank,
                "documents": len(assigned_docs),
                "windows": len(indices),
                "output": str(output_path),
                "summary": str(summary_path),
            })

    if total_windows != len(all_doc_ids):
        raise RuntimeError(f"validated {total_windows} windows, expected {len(all_doc_ids)}")

    result = {
        "status": "complete",
        "created_at_unix": time.time(),
        "prepared": fingerprint,
        "model": str(args.model),
        "model_label": args.model_label,
        "world_size": args.world_size,
        "documents": len(ordered_docs),
        "windows": total_windows,
        "target_length": target_length,
        "ranks": rank_details,
    }
    if args.write_success:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(success_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
