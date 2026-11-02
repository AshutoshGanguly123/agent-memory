# Reuse GPUs after conversation-local schema failures

User authorization: on 10 September 2026 the user requested skipping failed work and using the freed GPU for other data. The unit skipped is an entire failed conversation; a turn is never silently omitted from a continuing memory graph.

## Behavior

The original direct supervisor blocked a GPU after any unexpected nonzero worker exit. The separately versioned `operations/direct_reuse_v2` supervisor distinguishes a proven exhausted-schema conversation failure from an unknown or device/backend error.

A GPU is reusable only when the owned worker/exit/failure records agree and the conversation journal contains exactly the configured three completed schema-error attempts for the same logical call, with no success, backend error or context overflow. The failure marker, partial memory, call IDs, seeds, retries and all raw evidence remain intact. The unchanged worker's conversation-claim function skips failed or completed conversations and honors exclusive locks on active conversations.

After this validation, the supervisor may launch another unchanged worker on that GPU to claim different queued work. Normal UUID checks, minimum free VRAM/host RAM, per-GPU locks, node scheduling checks and cooperative release still apply. Unknown failures keep their GPU blocked pending inspection. A GPU may legitimately remain idle when there is no unclaimed eligible conversation.

No model, prompt, generation budget, scoring, inference method or within-conversation ordering changes. The 49 frozen original files and seven v1 adapter files remain frozen. The new supervisor reuses the original worker and backend; it has a separate source manifest and audit records.

## Handover and files

`operations/direct_reuse_v2/rollout.py` is a one-time migration from the original running supervisor. It runs the original and new software checks before signaling our verified supervisor with SIGUSR1. It waits for workers to stop at durable call boundaries, confirms owned exit records and absence of old workers, archives the raw run, records checkpoint/failure hashes, then starts the replacement in tmux `amem_direct_full_v2`.

The replacement holds the same legacy coordinator/worker locks. It accepts only the exact audited checkpointed handover state, so a second supervisor cannot overlap. The original direct-phase start timestamp is preserved for cost accounting; a separate supervisor-start timestamp records the transition. These commands are not a generic blind crash-retry loop.

Authoritative live state remains `results/operations/direct_parallel_v1/full_coordinator.json`, now with `supervisor_version=direct_reuse_v2`. Worker status, exits, failures, telemetry and conversation journals remain in their original locations. `conversation_skipped_gpu_released` events record the validation evidence behind reuse.

New deployment evidence lives under `results/operations/direct_reuse_v2/`:

- `manifest.json`: adapter source hashes and original manifest identities.
- `rollout.log`, `rollout.exit`, `launch.log`, `launcher_exit.json`: durable execution records.
- `migrations/<timestamp>/`: software checks, before/stop/checkpoint snapshots, original coordinator and raw backup.
- `migration_ready.json`: exact state accepted for this handover.
- `launch_verified.json`: new workers, physical GPU occupancy, advancing queued-conversation checkpoints and preserved failed conversations.

The global `results/operations/direct_parallel_v1/STOP` mechanism remains a durable full-stop request. Never erase failure markers or increase retry counts to enable reuse. If a later supervisor stop requires recovery, inspect and record a new compatible handover; do not blindly rerun the one-time rollout or the historical v1 full launcher.

## Reporting

Keep failed conversations and their original benchmark denominators visible. Processing the remaining eligible conversations does not make a run with missing conversations a complete 5,882-turn/1,986-question benchmark. Distinguish conversation failures from GPU faults and distinguish no eligible work from a resource failure.

Include the supervisor handover in elapsed-time and ingestion-cost reporting. The ingestion profiler continues to read the shared coordinator and records the preserved original phase start plus the supervisor metadata.
