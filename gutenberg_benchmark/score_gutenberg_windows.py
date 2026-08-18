#!/usr/bin/env python3
"""Score prepared Gutenberg windows with one causal language model on one GPU."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--loss-chunk", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Return successfully when this rank already has a valid checkpoint.",
    )
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
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
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rank_checkpoint_valid(
    output_path: Path,
    summary_path: Path,
    prepared: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    args: argparse.Namespace,
    target_length: int,
) -> bool:
    if not output_path.is_file() or not summary_path.is_file():
        return False
    try:
        with np.load(output_path, allow_pickle=False) as checkpoint:
            required = {
                "nll", "document_id", "language", "position", "anchor_char",
                "token_center", "target_start", "document_token_count",
            }
            if not required.issubset(checkpoint.files):
                return False
            nll = checkpoint["nll"]
            if nll.shape != (len(indices), target_length) or not np.isfinite(nll).all():
                return False
            for key in required - {"nll"}:
                if not np.array_equal(checkpoint[key], prepared[key][indices]):
                    return False
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_documents = len(set(int(value) for value in prepared["document_id"][indices]))
        return all(
            (
                summary.get("model") == str(args.model),
                summary.get("model_label") == args.model_label,
                summary.get("prepared") == str(args.prepared),
                summary.get("rank") == args.rank,
                summary.get("world_size") == args.world_size,
                summary.get("documents") == expected_documents,
                summary.get("windows") == len(indices),
                summary.get("target_length") == target_length,
            )
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def float32_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    pieces = []
    for start in range(0, targets.shape[1], chunk_size):
        end = min(start + chunk_size, targets.shape[1])
        chunk = logits[:, start:end, :].float()
        loss = F.cross_entropy(
            chunk.reshape(-1, chunk.shape[-1]),
            targets[:, start:end].reshape(-1),
            reduction="none",
        )
        pieces.append(loss.reshape(targets.shape[0], end - start))
    return torch.cat(pieces, dim=1)


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    overall_started = time.perf_counter()
    prepared = np.load(args.prepared, allow_pickle=False)
    all_doc_ids = prepared["document_id"]
    ordered_docs = list(dict.fromkeys(int(value) for value in all_doc_ids))
    assigned_docs = set(ordered_docs[args.rank :: args.world_size])
    indices = np.asarray(
        [i for i, value in enumerate(all_doc_ids) if int(value) in assigned_docs],
        dtype=np.int64,
    )
    if len(indices) == 0:
        raise RuntimeError(f"rank {args.rank} received no windows")

    input_ids_np = prepared["input_ids"][indices]
    context_length = int(prepared["context_length"])
    target_length = int(prepared["target_length"])
    expected_length = context_length + target_length
    if input_ids_np.shape[1] != expected_length:
        raise ValueError(f"prepared sequence length is {input_ids_np.shape[1]}, expected {expected_length}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_label}.rank{args.rank:02d}-of-{args.world_size:02d}"
    output_path = args.output_dir / f"{stem}.npz"
    summary_path = args.output_dir / f"{stem}.summary.json"
    if args.resume and rank_checkpoint_valid(
        output_path, summary_path, prepared, indices, args, target_length
    ):
        print(json.dumps({
            "model_label": args.model_label,
            "rank": args.rank,
            "status": "checkpoint_complete",
            "documents": len(assigned_docs),
            "windows": len(indices),
        }, indent=2))
        return

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model_load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    supports_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    torch.cuda.reset_peak_memory_stats(device)

    output_nll = np.empty((len(indices), target_length), dtype=np.float32)
    forward_seconds = 0.0
    loss_seconds = 0.0
    batch_seconds = []

    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(indices), args.batch_size)):
            end = min(start + args.batch_size, len(indices))
            ids = torch.as_tensor(input_ids_np[start:end], dtype=torch.long, device=device)
            torch.cuda.synchronize(device)
            forward_started = time.perf_counter()
            if supports_keep:
                outputs = model(
                    input_ids=ids,
                    use_cache=False,
                    logits_to_keep=target_length + 1,
                )
                prediction_logits = outputs.logits[:, :-1, :]
            else:
                outputs = model(input_ids=ids, use_cache=False)
                prediction_logits = outputs.logits[
                    :, context_length - 1 : context_length - 1 + target_length, :
                ]
            torch.cuda.synchronize(device)
            current_forward = time.perf_counter() - forward_started
            forward_seconds += current_forward

            targets = ids[:, context_length : context_length + target_length]
            loss_started = time.perf_counter()
            losses = float32_cross_entropy(prediction_logits, targets, args.loss_chunk)
            torch.cuda.synchronize(device)
            current_loss = time.perf_counter() - loss_started
            loss_seconds += current_loss
            output_nll[start:end] = losses.cpu().numpy()
            batch_seconds.append(
                {
                    "batch": batch_number,
                    "windows": end - start,
                    "forward_seconds": current_forward,
                    "loss_seconds": current_loss,
                }
            )
            del ids, outputs, prediction_logits, targets, losses

    if not np.isfinite(output_nll).all():
        raise RuntimeError("Non-finite surprisal values were produced")

    save_started = time.perf_counter()
    atomic_savez(
        output_path,
        nll=output_nll,
        document_id=prepared["document_id"][indices],
        language=prepared["language"][indices],
        position=prepared["position"][indices],
        anchor_char=prepared["anchor_char"][indices],
        token_center=prepared["token_center"][indices],
        target_start=prepared["target_start"][indices],
        document_token_count=prepared["document_token_count"][indices],
    )
    save_seconds = time.perf_counter() - save_started
    input_tokens = int(len(indices) * expected_length)
    target_tokens = int(len(indices) * target_length)
    gpu_compute_seconds = forward_seconds + loss_seconds
    summary = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": args.rank,
        "world_size": args.world_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "model": str(args.model),
        "model_label": args.model_label,
        "parameter_count": parameter_count,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "supports_logits_to_keep": supports_keep,
        "prepared": str(args.prepared),
        "output": str(output_path),
        "batch_size": args.batch_size,
        "documents": len(assigned_docs),
        "windows": len(indices),
        "context_length": context_length,
        "target_length": target_length,
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "model_load_seconds": model_load_seconds,
        "forward_seconds": forward_seconds,
        "loss_seconds": loss_seconds,
        "gpu_compute_seconds": gpu_compute_seconds,
        "input_tokens_per_second": input_tokens / gpu_compute_seconds,
        "target_tokens_per_second": target_tokens / gpu_compute_seconds,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "save_seconds": save_seconds,
        "elapsed_seconds": time.perf_counter() - overall_started,
        "batches": batch_seconds,
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in (
        "model_label", "rank", "documents", "windows", "batch_size",
        "model_load_seconds", "gpu_compute_seconds", "input_tokens_per_second",
        "target_tokens_per_second", "peak_gpu_memory_gb", "elapsed_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
