# HPC A-Mem experiment

- Canonical root: `${REPO_ROOT}/a-mem-locomo` on `<hpc-host>`. The Mac `agent_memory` tree is for source editing and review.
- Read `docs/PRD_TT_AS_BUILT.md` and `docs/HPC_RUNBOOK.md`. Scope: one pinned Llama-3.2-3B-Instruct model, A-Mem, all 5,882 LoCoMo turns and 1,986 questions.
- Keep TT separate. Never restart TT or transfer its models, dataset, checkpoints or results here. Download assets independently from pinned original sources.
- All setup, tests, inference and analysis run in tmux with `.venv`, durable logs and exit files. The user's 2026-09-10 instruction replaces Slurm execution with direct SSH on <gpu-node>. Do not submit or restart Slurm jobs. Read docs/DIRECT_PARALLEL_RUNBOOK.md and operations/direct_parallel_v1. No CPU LLM fallback; CPU embeddings/tokenization/scoring are allowed.
- Use the explicitly authorized direct-node launcher on physically idle cards, with GPU UUID checks, sufficient memory, per-GPU/conversation locks and cooperative release when competing physical or scheduled use appears. Never reset GPUs or kill another user's processes. Preserve conversation isolation and chronological writes; freeze before QA.
- Never ingest questions, gold answers, distractors, evidence, summaries, images or captions into memory.
- Freeze protocol/code/prompts/assets/environment before full inference. New scientific changes need new run IDs; operational fixes require versioned audit records.
- Retain full raw calls, failures, seeds, retries and overflow; never truncate prompts silently or remove failed questions from denominators. Preserve stale embedding state on resume.
- Update `results/further_work/observations.md` with dated measured observations and separately labeled untested hypotheses; preserve user edits.
- Report partial progress honestly. Finish six metrics, four figures and an official NeurIPS-format report; inspect the PDF before declaring completion. No automatic publication.
- Do not spawn subagents unless the user explicitly requests it.

## Current node scope (user instruction, 11 September 2026)

- The user expanded direct SSH execution from <gpu-node> to available `gpu-xx` nodes. Use `sinfo` plus physical occupancy and detailed Slurm GPU-index reservations. Never SSH into `ws` nodes. Do not submit Slurm jobs.
- The active four-conversation 2,048-token recovery uses `operations/recovery_gpu_nodes_v3`; its coordinator is `results/operations/recovery_gpu_nodes_v3/coordinator.json`. Worker journals/checkpoints remain in the existing recovery paths. Read `docs/RECOVERY_OUTPUT2048_RUNBOOK.md` for the audited handover. Frozen earlier adapters remain historical.
