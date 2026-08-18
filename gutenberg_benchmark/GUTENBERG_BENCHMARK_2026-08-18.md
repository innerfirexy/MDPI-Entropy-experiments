# Gutenberg spectrum-validation benchmark (2026-08-18)

## Scope

- Host: `a5880-1`
- GPUs: 8 x NVIDIA RTX 5880 Ada Generation (48 GB)
- Models:
  - `/nas/model/Llama3.1/Llama3.1-8B`
  - `/nas/model/qwen3/Qwen3-8B-Base`
- Input per document: three text windows (early, middle, late)
- Input per window: 1024 context tokens + 4096 scored target tokens
- Samples: 100 documents (single GPU) and 800 documents (8 GPUs)
- Languages in the balanced benchmark: `de`, `en`, `es`, `fi`, `fr`, `hu`, `it`, `nl`, `pt`, `zh`
- Batch size: 4 windows per GPU
- Runtime condition: another user's evaluation occupied about 2.1--2.3 GB and about 30% utilization on every GPU. The measurements below are shared-load measurements, not exclusive-machine peak throughput.

## Correctness checks

- 100-document runs: 100 documents and 300 windows per model.
- 800-document runs: 800 documents and 2400 windows per model, split evenly as 100 documents per GPU rank.
- Two 800-document runs produced 16 NPZ shards containing 19,660,800 NLL values; all values were finite.
- No traceback, CUDA OOM, or runtime error was found in the 8-GPU logs.
- The diagnostic analyzer merged all shards and recovered all 10 languages and all ten octave bands.
- The shuffle analyses used 200 permutations. They are benchmark diagnostics, not confirmatory corpus results.

## Measured scoring performance

| Model | Run | Documents | Slowest-rank compute | Slowest-rank wall time | Aggregate target throughput | Peak model-process GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B Base | 1 GPU | 100 | 301.6 s | 497.3 s | 4,075 target tok/s | 21.32 GB |
| Qwen3-8B Base | 1 GPU | 100 | 322.0 s | 488.0 s | 3,816 target tok/s | 22.62 GB |
| Llama-3.1-8B Base | 8 GPUs | 800 | 300.9 s | 353.3 s | 32,666 target tok/s | 21.32 GB/rank |
| Qwen3-8B Base | 8 GPUs | 800 | 321.8 s | 363.0 s | 30,550 target tok/s | 22.62 GB/rank |

The 8-GPU aggregate compute throughput is approximately 8.02x (Llama) and 8.01x (Qwen) the corresponding single-GPU throughput. Model cold-load time was 196/166 s for the first single-GPU loads and 52/41 s for the cached 8-GPU loads.

## Full-corpus projection

The corpus currently contains 69,949 text files. Treating every file as eligible for three complete windows gives an upper-bound workload of 859,533,312 scored target tokens (1,074,416,640 input tokens including context).

At the measured shared-load 8-GPU compute throughput:

- Llama-3.1-8B Base: about 7.31 GPU-compute hours, or 7.32 hours including one cached model load.
- Qwen3-8B Base: about 7.82 GPU-compute hours, or 7.83 hours including one cached model load.

CPU preparation scaled linearly from the 800-document run to about 6.95 hours per model. The current single-process 200-shuffle analyzer projects to about 1.96 hours for Llama and 1.85 hours for Qwen. If preparation, scoring, and analysis are run strictly serially, the upper-bound end-to-end estimates are therefore about 16.23 hours (Llama) and 16.61 hours (Qwen). Preparation and analysis can overlap GPU scoring, so calendar time can be lower.

These are conservative upper bounds because short files that cannot supply three non-overlapping windows must be excluded. A full-corpus eligibility manifest should be generated before the confirmatory run.

## Output locations

- Code and manifests: `/home/xy/projects/MDPI-Entropy-experiments/gutenberg_benchmark/`
- Prepared inputs: `/home/xy/projects/MDPI-Entropy-experiments/results/gutenberg_benchmark/prepared/`
- 100-document results: `/home/xy/projects/MDPI-Entropy-experiments/results/gutenberg_benchmark/sample100_single_gpu/`
- 800-document results: `/home/xy/projects/MDPI-Entropy-experiments/results/gutenberg_benchmark/sample800_8gpu/`
- Single-GPU logs: `/home/xy/projects/MDPI-Entropy-experiments/logs/gutenberg_benchmark/`
