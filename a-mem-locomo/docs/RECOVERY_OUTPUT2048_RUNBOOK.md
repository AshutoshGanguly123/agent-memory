# Four-conversation recovery with a 2,048-token output cap

Authorized by the user on 11 September 2026: recover only conv-26, conv-41, conv-44 and conv-50 with the higher output limit. Do not execute ingestion or QA again for the six completed conversations, including conv-43's preserved failed QA record.

Run ID: `locomo_llama32_3b_amem_cuda_output2048_recovery_v1`.


## Active adapter v3: validated replay on <gpu-node>

The active launcher is now operations/recovery_gpu_nodes_v3/start.py, tmux amem_recovery_nodes_v3_run, with coordinator results/operations/recovery_gpu_nodes_v3/coordinator.json. It started on <gpu-node> at 2026-09-11 10:44:57 UTC. Initial active cards are 6 and 7; the other six physical cards were occupied. The queue remains exactly conv-41, conv-50, conv-26, conv-44. The supervisor can only use cards listed in its validation.json on this node. Workers, STOP, failure records and raw output retain the v1 recovery paths described below.

Historical-token replay in v2 produced valid 89-token JSON on four <gpu-node> cards, all mutually identical, but differed from the historical 81-token conv-41 note on <gpu-node>. The first differing token is within the context description. That failed strict replay is preserved. On <gpu-node>, a further 1,000-cap replay and two 2,048-cap replays produced exactly the same 89 tokens and parsed JSON; they also matched all four initial node probes. This establishes the observed difference is present at both caps and stable for these probes. It does not establish the underlying cause or bit-identical reproduction of historical <gpu-node> calls.

The separately frozen v3 validation requires valid JSON and exact replay of this documented <gpu-node> reference on every selected card, plus verified cap/repeat stability and unchanged frozen scientific code. It records historical_token_parity=false explicitly. Cards 6 and 7 passed; 21 scheduling/reference tests passed. A new node is an execution provenance change in the already separate recovery run. Do not describe the node migration as bit-identical historical reproduction. No inference policy or recovery parameter was changed to obtain a passing result.

Evidence: v2/probes, replay_gpu3_diagnosis.json/log/exit, the unchanged v2 manifest, and v3/{manifest.json,validation.json,checks.log,selection.json,handover.json}. The old idle <gpu-node> supervisor PID 921412 cooperatively exited after SIGUSR1 at 10:35:56 UTC, with zero children and checkpointed state. All original baseline artifacts and inherited lineage were checked before deployment. User lingering was enabled for ashutosh.ganguly on <gpu-node> (previously no) to retain the user service manager after SSH logout; no system-wide or other-user setting was changed.

## Initial node adapter (v2): available gpu-xx nodes

On 11 September the user explicitly expanded direct-node execution beyond <gpu-node> to available `gpu-xx` nodes and excluded SSH into `ws` nodes. The versioned `operations/recovery_gpu_nodes_v2` overlay keeps the frozen recovery configuration, original inference methods, scope, journals and checkpoints. No scientific parameters change. The earlier <gpu-node>-only instructions below are historical.

The replacement is prepared on <gpu-node> through `.venv/bin/python operations/recovery_gpu_nodes_v2/start.py prepare` in tmux `amem_recovery_nodes_prepare`. It runs scheduling tests, freezes the overlay, and validates up to four idle, unreserved RTX 5000 Ada cards by exactly replaying the token IDs and JSON of an inherited successful conv-41 note. These diagnostic replays do not ingest any turns, answer questions, or modify benchmark journals. Probe compute is logged separately.

After a verified cooperative stop of the old waiting supervisor, `.venv/bin/python operations/recovery_gpu_nodes_v2/start.py run` starts tmux `amem_recovery_nodes_run` on the validated node. The single global coordinator lock prevents competing supervisors. New authoritative coordinator state, validation, source hashes and handover are under `results/operations/recovery_gpu_nodes_v2/`. Worker status, exits, logs, locks, failure markers, telemetry and STOP remain under `results/operations/recovery_output2048_v1/`. Both earlier coordinator records are historical after handover.

Availability requires physical occupancy checks and `scontrol show node -d` GPU index mappings; an apparently idle card may still be reserved. A scheduled allocation on another card no longer stops every worker on a mixed node. Reservations or competing processes on a used card request a durable stop of our corresponding worker only. Unknown scheduler mappings, query failures and unavailable/drained nodes block launches and request cooperative stops. GPU UUID validation, free-memory thresholds, per-GPU locks and 15-minute cooldowns remain. Only validated cards on the selected node can launch; this is not an automatic multi-node migration service.

## Scientific scope and lineage

The original run and its configuration remain unchanged. The recovery uses the original frozen worker, CUDA backend, prompts, memory logic, dataset and scoring. Its separate configuration changes only the run ID and `max_new_tokens` from 1,000 to 2,048, for all new model calls in the four selected conversations. The new configuration changes derived call IDs and seeds. The maximum three attempts persists across restarts within the recovery configuration.

This is a checkpoint-derived recovery, not a fresh experiment using 2,048 tokens throughout. It inherits 609 notes built with the original cap: conv-41 54, conv-50 91, conv-26 47 and conv-44 417. Exactly copied checkpoints preserve note order, links, stale embeddings, index documents and evolution counters. The four conversations have 2,325 notes in total, so 1,716 remain to ingest, followed by 754 QA questions. The conversation order prioritizes the most remaining ingestion: 41, 50, 26, 44.

Each selected parent's entire conversation folder is copied under the recovery's `lineage/conversations/`. The active conversation folders start with copies of `memory.json` and `memory_events.jsonl`; their new `calls.jsonl` journals are separate. This retains every parent call, failed attempt and seed without reusing an exhausted old-config call budget. A successful but uncommitted parent note call at the failure boundary is regenerated with the new configuration. Previously committed turns are skipped using the copied checkpoint. Nothing is deleted from the original run.

Interpret combined results with the six baseline conversations as mixed-policy recovery results. Preserve both the baseline ingestion failures and the recovery outcomes. Do not silently report this as a complete uniform-cap benchmark, discard missing questions, or count baseline cost twice because copies exist in lineage.

## Preparation and execution

Code: `operations/recovery_output2048_v1/`. Operations state: `results/operations/recovery_output2048_v1/`. Raw output: `results/raw_results/locomo_llama32_3b_amem_cuda_output2048_recovery_v1/`.

Preparation runs original software checks and new recovery tests in tmux with `.venv`. It verifies the 49 original source/config files, seven original direct adapter files, the existing GPU-reuse adapter, and all baseline raw artifacts. It copies only the selected conversations, verifies the copied data, freezes its own source/config/manifests, and writes `ready.json`. Originals are checked again after copying and at recovery startup/completion.

From <gpu-node>, use the project `.venv` to run `operations/recovery_output2048_v1/start.py run` after preparation passes. This starts `amem_recovery_2048_run`. The supervisor waits for physically available GPUs and can start with one; it uses at most four workers. Each worker has a hard conversation-scope guard. It shares the legacy coordinator/GPU locks and retains original UUID/occupancy checks, at least 28 GiB free VRAM, at least 32 GiB available host RAM, and cooperative stops for competing physical or scheduled work. No Slurm jobs are submitted. No foreign process is signaled or reset.

`state=waiting_for_gpus` means the durable supervisor is active but no inference is running. `state=running` requires worker processes. Read `coordinator.json`, `workers/*.json`, `events.jsonl`, `telemetry.jsonl` and raw checkpoints for current progress. The original coordinator is historical and must not be mistaken for recovery state.

## Failures and stopping

Schema exhaustion is classified using the already validated GPU-reuse policy against the recovery's new journals. It preserves the failed recovery conversation and allows a free GPU to take another queued target. Unknown/backend errors block the affected card and remain visible. No fourth attempt or automatic further cap increase is allowed.

Creating `results/operations/recovery_output2048_v1/STOP` requests a durable stop of this recovery. Startup refuses an existing recovery coordinator state; an unexpected supervisor exit needs inspection and an audited resumption, not blind relaunch. The preparation and run launchers write separate exit records. A successful four-conversation run does not invoke the frozen full-dataset analyzer; a separate analysis must account for inherited prefixes and the explicit four-conversation scope.
