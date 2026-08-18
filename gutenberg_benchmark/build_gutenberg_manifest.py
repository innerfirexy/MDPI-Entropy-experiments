#!/usr/bin/env python3
"""Build a deterministic manifest from a Gutenberg catalog and local corpus."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import TextIO


OUTPUT_FIELDS = (
    "document_id",
    "language",
    "file_size_bytes",
    "catalog_type",
    "catalog_issued",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-snapshot-date", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-catalog",
        action="store_true",
        help="Retain corpus IDs absent from the catalog with language=und.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_catalog(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def load_catalog(path: Path) -> dict[int, dict[str, str]]:
    with open_catalog(path) as handle:
        reader = csv.DictReader(handle)
        required = {"Text#", "Language", "Type", "Issued"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"catalog must contain {sorted(required)}")
        catalog: dict[int, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            raw_id = (row.get("Text#") or "").strip()
            if not raw_id.isdigit():
                raise ValueError(f"invalid Text# at catalog row {row_number}: {raw_id!r}")
            document_id = int(raw_id)
            if document_id in catalog:
                raise ValueError(f"duplicate catalog Text#: {document_id}")
            catalog[document_id] = row
    return catalog


def load_corpus(corpus_dir: Path) -> dict[int, int]:
    if not corpus_dir.is_dir():
        raise ValueError(f"corpus directory does not exist: {corpus_dir}")
    files: dict[int, int] = {}
    nonnumeric = []
    for path in corpus_dir.glob("*.txt"):
        if not path.stem.isdigit():
            nonnumeric.append(path.name)
            continue
        document_id = int(path.stem)
        if document_id in files:
            raise ValueError(f"duplicate numeric document ID: {document_id}")
        files[document_id] = path.stat().st_size
    if nonnumeric:
        raise ValueError(f"nonnumeric corpus filenames: {sorted(nonnumeric)[:10]}")
    if not files:
        raise ValueError(f"no numeric .txt files found in {corpus_dir}")
    return files


def atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    corpus = load_corpus(args.corpus_dir)
    missing = sorted(set(corpus) - set(catalog))
    if missing and not args.allow_missing_catalog:
        raise RuntimeError(
            f"{len(missing)} corpus documents are absent from the catalog: {missing[:20]}"
        )

    rows = []
    for document_id, file_size in sorted(corpus.items()):
        metadata = catalog.get(document_id, {})
        language = (metadata.get("Language") or "und").strip() or "und"
        rows.append({
            "document_id": document_id,
            "language": language,
            "file_size_bytes": file_size,
            "catalog_type": (metadata.get("Type") or "unknown").strip() or "unknown",
            "catalog_issued": (metadata.get("Issued") or "").strip(),
        })

    atomic_write_csv(args.output, rows)
    inventory_text = "".join(
        f"{document_id},{file_size}\n" for document_id, file_size in sorted(corpus.items())
    )
    metadata = {
        "catalog_file": args.catalog.name,
        "catalog_sha256": sha256_file(args.catalog),
        "catalog_snapshot_date": args.catalog_snapshot_date,
        "catalog_rows": len(catalog),
        "corpus_files": len(corpus),
        "corpus_inventory_sha256": hashlib.sha256(inventory_text.encode()).hexdigest(),
        "manifest_file": args.output.name,
        "manifest_sha256": sha256_file(args.output),
        "missing_catalog_documents": missing,
        "language_label_counts": dict(sorted(Counter(row["language"] for row in rows).items())),
        "catalog_type_counts": dict(sorted(Counter(row["catalog_type"] for row in rows).items())),
    }
    metadata_path = args.output.with_suffix(".meta.json")
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({
        "manifest": str(args.output),
        "documents": len(rows),
        "language_labels": len(metadata["language_label_counts"]),
        "catalog_types": metadata["catalog_type_counts"],
        "manifest_sha256": metadata["manifest_sha256"],
        "metadata": str(metadata_path),
    }, indent=2))


if __name__ == "__main__":
    main()
