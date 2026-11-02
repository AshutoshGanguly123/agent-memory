# A-Mem on LoCoMo — NVIDIA HPC reproduction

One Llama-3.2-3B-Instruct model, one complete LoCoMo benchmark, A-Mem, six primary metrics. This is a fresh CUDA experiment; TT data and results stay separate.

Canonical directory: `${REPO_ROOT}/a-mem-locomo` on SSH alias `<hpc-host>`.

That path links to the active project on HPC home storage because `/l` writes failed and stalled during setup. See the runbook's storage note. No TT assets are used.

Read [the audited TT PRD](docs/PRD_TT_AS_BUILT.md) for the complete scientific protocol and [the HPC runbook](docs/HPC_RUNBOOK.md) for operations. The schema and author prompts are in `configs/`. Raw calls and per-turn/per-question checkpoints go in `results/raw_results/<run_id>/`; analysis goes in `results/insights/<run_id>/`; add observations and research ideas to `results/further_work/observations.md`.

## Workflow

Run these from this repository, each in tmux. GPU work is submitted through Slurm by the launcher; do not invoke the CUDA worker on the login node.

```bash
python3 scripts/start.py setup
python3 scripts/start.py checks
python3 scripts/start.py preflight
python3 scripts/start.py full
```

These are sequential gates: wait for each successful exit before the next. Setup independently downloads pinned original assets and hashes them. Tests check source/scoring parity and restart behavior. Slurm preflight exercises all stages and the actual 131,072-token CUDA capacity. Full launch freezes sources and assets, then schedules independent shards within the one-GPU account quota. Resume uses the same full launcher; a coordinator lock prevents duplicate launchers. Unknown failures require inspection, not automatic repeated retries.

The launch currently prefers `<gpu-node>`. An actual allocated-device occupancy check protects against GPUs occupied outside Slurm's accounting. Busy devices are released and checked again after 15 minutes; other users' processes are left alone.

No model weights, dataset or raw results are committed to Git. Preserve downloaded upstream license files. Access to Meta's gated model uses the existing HPC Hugging Face authentication without storing credentials in this repository. This project is an experimental adaptation, not an exact numerical reproduction of paper scores.

## Metrics

1. Category-specific answer F1 / adversarial abstention accuracy.
2. Retrieved gold-evidence coverage and all-evidence rate.
3. Memory build and QA latency, with operational time separated.
4. Input/output tokens by phase, including retries.
5. Semantic memory footprint and directed link counts.
6. Context overflow by phase and context utilization.

Analysis generates four diagnostic views, per-question traces and failure cases. The final report uses the official NeurIPS 2026 style. Partial runs are always labeled partial; PDF visual review remains necessary before declaring the report final.
