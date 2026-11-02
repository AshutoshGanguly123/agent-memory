# A-Mem × LoCoMo: TT implementation audit and NVIDIA reproduction requirements

Owner: Ashutosh Ganguly. Audit date: 10 September 2026 (UTC). Status: implementation specification, not completed benchmark results.

## 1. Objective and scope

Run one open-weight language model with A-Mem on the complete LoCoMo benchmark, retain enough evidence to locate individual failures, and develop research hypotheses from measured behavior. The experiment is Llama-3.2-3B-Instruct, not the earlier proposed Qwen models. No training, extra benchmarks, model sweep, no-memory baseline, or publication is included. A single run cannot establish that A-Mem improves over a baseline, that a failure has a particular cause, or that an idea is publishable.

The NVIDIA implementation must preserve the TT experiment's scientific protocol while replacing its native accelerator implementation and resource scheduling. It is a fresh run, with separately obtained assets and a new run ID. TT checkpoints, datasets, weights and results must not be transferred to HPC or incorporated into its results. Existing experiment source may be deployed from the Mac development repository; external assets must come directly from their original pinned sources or verified HPC-local caches.

## 2. Audited deployment and current status

- TT host: `tt-blackhole-02`, SSH alias `Tenstorrent`.
- TT root: `/home/ashutoshganguly/agentic_memory/A_Mem_reproduction`.
- Hardware: four independent Tenstorrent Blackhole P150b cards. Native model forwards, no CPU fallback. CPU handles MiniLM, tokenization, constrained sampling and scoring.
- TT backend checkout: `/home/ashutoshganguly/watermarking_TT_vendor/tt-metal-32f12ad`, commit `32f12adde09c3596415982cfdbfd8e09f5ccde73`.
- Original run: `locomo_llama32_3b_amem_v1`. Saved progress: 732/5,882 notes, 0/1,986 questions. Conversation counts: conv-48 218; conv-47 136; conv-43 147; conv-44 231.
- At 12:27:30 UTC the owned idle scheduler PID 535291 was terminated after verifying it had no worker children. The stop is recorded under `results/operations/shared_scheduler_20260910/stopped_for_hpc_migration.json`. TT artifacts remain intact. Do not restart this experiment during the HPC run.

Read-only evidence inspected on TT: `AGENTS.md`, `README.md`, `configs/experiment.json`, schemas and prompts, `src/amem_reproduction/{memory,prompts,calls,runner,data,scoring,native,storage}.py`, asset manifests, original launch implementation, recovery and shared-scheduler overlays, and current process/status records. This document describes the code actually built, including deviations from the paper and released implementation.

## 3. Pinned scientific inputs

| Asset | Identity |
|---|---|
| Generator | `meta-llama/Llama-3.2-3B-Instruct` |
| Model revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| A-Mem source | `WujiangXu/A-mem`, commit `c3bb1ee6d4825b13460dd26be1444b5f870ddb64` |
| LoCoMo source | `snap-research/locomo`, commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, `data/locomo10.json` |
| Dataset SHA-256 | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| Model shard 1 SHA-256 | `13cbd6d16e927a0c5bad54102514e6e18b4a47b3a6eb911e39d678d328d19f55` |
| Model shard 2 SHA-256 | `7b770216613ac5c34d7c54bdff1fa616bc4e338a9d0b20af6303e48c295ee23c` |

LoCoMo has ten conversations, 5,882 text turns and 1,986 questions: 282 multi-hop, 321 temporal, 96 open-domain, 841 single-hop and 446 adversarial. It is both a dataset and an evaluation benchmark when paired with a task/protocol/scorer.

Sessions are ordered numerically, then turns retain their original order. Memory input is only the turn text, speaker, dialogue ID, session identifier and session date/time. The adapter retains a caption-presence diagnostic, but captions, images, supplied summaries, questions, gold answers, adversarial distractors and evidence annotations never enter memory construction. Image-dependent questions remain in the denominator; this is explicitly a text-only evaluation.

## 4. System design and triggering rules

```text
Pinned LoCoMo → chronological conversation adapter
  → per-conversation memory builder
      → LLM note enrichment
      → MiniLM top-5 existing-note search
      → LLM evolution decision → update notes/links
      → embed new note; periodically refresh existing index
      → durable call log + atomic memory checkpoint after each turn
  → freeze conversation memory and fingerprint it
  → for every question: LLM keyword rewrite
      → MiniLM top-10 search + released directed-link expansion
      → LLM short answer → released LoCoMo scorer
      → durable answer record, including failures and retrieval provenance
  → offline metrics → four plots → failure-case CSV → NeurIPS-format report
```

Each logical worker owns its own conversations and never shares mutable memory. TT's four logical shards are 0: conv-48, conv-49, conv-26; 1: conv-47, conv-50, conv-30; 2: conv-43, conv-42; 3: conv-44, conv-41. Parallelism is across conversations; chronological writes within a conversation remain serial. The same language model performs all four LLM roles. There is no separate graph database, autonomous background evolution, online training, or persistent chat-session state inside the model.

### Note creation

Each new turn triggers enrichment with the author-derived prompt. Content formatting is exactly `Speaker ` + speaker + `says : ` + text (including the source spacing). Output is a JSON object containing `keywords: string[]`, `context: string`, and `tags: string[]`. Empty context becomes `General`. The new note then searches existing vector embeddings using raw formatted content, retrieving at most five candidates.

### Evolution and edges

Every insertion, including the first note with no neighbors, triggers the evolution prompt. Its JSON requires `should_evolve`, `actions`, `suggested_connections`, `tags_to_update`, `new_context_neighborhood`, and `new_tags_neighborhood`. Actions are restricted to `strengthen` and `update_neighbor`; suggested integer links are restricted to the retrieved candidates. These validity constraints are a documented adaptation.

When `should_evolve` is false, none of the proposed actions apply. When true, `strengthen` appends candidate insertion indices to the NEW note's `links` and replaces the NEW note's tags. `update_neighbor` replaces OLD neighbors' tags/context in retrieval order. Raw content and keywords remain unchanged. The released partial-array loop is preserved: tag-array length bounds which neighbors are visited, and context is replaced only where an entry exists. Short arrays are diagnosed, not silently repaired.

Edges are directed integer references in the originating note's `links` array. There is no edge label, weight, explanation object or automatic reverse edge. Duplicates are possible. The unused `evolution_history` field is initialized empty; actual before/after history is in the external event journal.

### Embedding freshness

The new note is immediately embedded from its content, context, keywords and tags. Updated older metadata does NOT immediately update old vectors. A positive evolution decision increments a counter. Every 100 positive decisions, the whole index is recomputed, using the released consolidation formatting. Both evolving notes and the possibly stale index documents/vectors must be checkpointed. Re-embedding everything on resume would change the experiment.

### Retrieval and answering

Only after all turns are ingested does memory freeze. Each question triggers keyword generation (the prompt requests `cosmos` separation while its example uses commas; preserve this inconsistency). Search returns ten primary notes. Directed links are expanded using the released global counter and append-before-cap behavior, which can include duplicates and exceed a naive ten-link cap. Do not deduplicate the supplied context as an optimization. Save primary and expanded indices and IDs.

Temporal questions use the released date-aware prompt. Other questions request short phrases with exact context wording. Category 5 additionally requests `No information available` for absent information and never receives the dataset distractor. This generative abstention protocol differs from the author's multiple-choice adversarial path. QA cannot update memory; its fingerprint must remain unchanged.

## 5. Note and persistence schemas

Illustrative note (values shown are examples, not measured outputs):

```json
{
  "id": "D2:3", "content": "Speaker Mayasays : I moved to Paris.",
  "timestamp": "10:00 am on 1 January, 2025", "speaker": "Maya",
  "session": "session_2", "keywords": ["Maya", "Paris"],
  "context": "Maya describes moving to Paris", "tags": ["location"],
  "links": [0, 2], "importance_score": 1.0, "retrieval_count": 0,
  "last_accessed": "10:00 am on 1 January, 2025",
  "evolution_history": [], "category": "Uncategorized"
}
```

The `memory.json` checkpoint contains `notes`, `index_documents`, `embeddings`, `evolution_count`, `frozen`, and accumulated `build_seconds`. Index positions are stable. A fingerprint excludes only elapsed build time. `memory_events.jsonl` records insertion IDs, note/evolution call IDs, retrieved neighbors, raw decision, before/after neighbors, new note, refresh flag, partial-array flag, latency and footprint. Its deterministic event ID supports deduplication if a crash occurs after journaling but before checkpoint commit.

`calls.jsonl` records full prompts, system prompt, schemas, logical/call ID, input count, output reserve, phase, attempt, seed, start/end timestamps, raw response, generated token IDs/count, parsed result, validation status, stop reason, prefill and total latency, overflow and backend errors. Successful cached calls are reused after an interrupted turn. The recovery overlay preserves the three-complete-attempt schema retry cap across restarts. Operationally interrupted attempts can be retried at the same seed and are recorded separately.

`answers.jsonl` includes question/category/gold, prediction, score, status/error, rewritten query, retrieval indices/IDs/duplicates, evidence hits/invalid IDs/coverage, call IDs, frozen fingerprint and latency. Gold/evidence are read for scoring and diagnostics only. Failed QA stays in the denominator with zero score. Missing evidence coverage is null. `complete.json` is written only when that conversation's notes and all questions are done.

Atomic checkpoint replacement and fsync journal writes protect restart state. A trailing incomplete JSONL line is archived before repair; corruption in the middle must raise. Conversation/worker locks prevent competing writers. A hard kill can lose the current incomplete generation, but must not lose completed calls or earlier turn checkpoints. Never count unknown crash output as known zero.

## 6. Scientific and decoding parameters

| Parameter | Value |
|---|---|
| Maximum total context | 131,072 tokens |
| Output reserve / generation cap | 1,000 tokens per request |
| Temperature | 0.7 |
| Seed | 42; deterministic request/attempt derivation |
| Complete schema attempts | 3 across restarts |
| Write/read neighbors | 5 / 10 |
| Index refresh | Every 100 positive evolution decisions |
| Batch size | 1 per independent worker |
| JSON enforcement | lm-format-enforcer 0.11.3 + JSON Schema validation |
| Stop | Complete schema-valid JSON, EOS, output cap or no valid tokens |
| TT prefill block | 4,096 tokens |
| TT precision | Native mixed BFP8/BF16 |
| MiniLM sequence cap | Model's 256 embedding tokens; distinct from LLM overflow |

The chat template has a system message `You must respond with a JSON object.` and one user prompt. Sampling is over allowed tokens with softmax temperature; no additional top-p/top-k/repetition policy is injected. The call digest includes logical ID, prompt, schema and complete config. Attempt seed is `(42 + int(sha256_json([call_id, attempt])[:8], 16)) % 2**31`. Consequently a new config/run ID changes derived seeds. Fixed policy is not identical TT/CUDA outputs. TT did not pin Python hash ordering; HPC should pin it for repeatability and disclose this change.

Input plus 1,000 reserve exceeding 131,072 is an explicit overflow. No hidden prompt truncation or alternative summarization. A memory-stage schema/overflow error stops the shard; a QA-stage error produces a failed record with score zero. Native/CUDA exceptions require investigation before resubmission.

## 7. Six primary metrics and four diagnostic views

1. **Answer quality:** released LoCoMo stemmed token F1 for categories 1–4, category-specific multi-hop averaging and open-domain answer handling; category 5 abstention-string accuracy separately. Report counts and failures. Do not blend F1 and abstention into an unexplained single score.
2. **Evidence coverage:** unique gold dialogue IDs present in the actual expanded answer context; fraction and all-evidence rate. Preserve invalid evidence IDs in the denominator and flag them. Missing annotations are null.
3. **Latency:** cumulative memory-build worker time and QA median/p95; separate prefill/decode, model load, checkpoints, restarts, wait/queue and end-to-end elapsed time. Cached retries can lower reconstruction timing; full call logs supply actual compute accounting.
4. **Tokens:** input/output by note, evolution, rewrite and answer, including retries; unknown crash outputs remain unknown.
5. **Memory footprint:** notes, directed links, UTF-8 note/index JSON bytes and float32 embedding bytes. Exclude weights, KV cache and trace archives from semantic memory footprint; expose checkpoint/log sizes separately.
6. **Context overflow:** failures by phase plus max prompt and p95 `(input + reserve) / window`. GPU OOM and embedding truncation are distinct diagnostics.

Operational diagnostics include schema failures, device/backend errors, GPU memory/utilization/power, CPU and host RAM. They support interpretation rather than expand the primary metric list.

Produce: (a) category × conversation quality heatmap with counts; (b) evidence coverage versus quality; (c) evidence age/span versus quality within category; (d) memory size and cumulative build cost growth. Export per-question traces and low-score `failure_cases.csv` with raw-trace pointers. Semantic review must precede claims about low lexical scores or a research gap.

## 8. Directory contract

```text
A_Mem_reproduction/
  AGENTS.md, README.md, pyproject.toml, requirements.lock, .gitignore
  .venv/                         isolated environment (ignored)
  docs/                          this PRD, HPC design, runbook
  configs/                       pinned experiment, schemas, exact prompts
  models/{weights,embeddings}/   independently sourced assets (ignored)
  models/manifest.json           identities and verified file hashes
  dataset/locomo10.json          fresh original-source download
  dataset/manifest.json          source/hash/counts
  benchmark/upstream/            pinned author source and licenses
  benchmark/report_template/     official NeurIPS style and source identity
  src/amem_reproduction/          memory, calls, backend, runner, scoring, analysis
  scripts/                       setup, verification, Slurm launch, status, report
  tests/                         source-parity and failure/restart tests
  results/
    setup/                       environment and verification logs
    operations/                  scheduler records and resource telemetry
    preflight/                   synthetic verification, separate from benchmark
    raw_results/<run_id>/         frozen manifest/source snapshot, conversations
    insights/<run_id>/            metrics, CSVs, visualisations, report
    further_work/observations.md dated observations and untested ideas
  tools/                         optional local report compiler
```

## 9. Operational history and lessons

The first TT full attempt stopped at 512 notes: three SIGKILL exits were observed, but their original cause was not established; another worker reached very high host RSS and was stopped. An operational overlay introduced garbage collection/malloc trimming, recycling after 30 durable calls or 6 GiB host RSS, and 12 GiB/300-second-progress/600-second-call watchdogs. Fourteen clean recycles advanced to 732 notes.

After other users occupied cards between recycles, two workers visibly waited for device locks; the other two stalled for an unestablished reason. Our own watchdog subsequently escalated to SIGKILL. Unprivileged `fuser` failed to expose other users' holders, so a later scheduler used `/proc/driver/tenstorrent/*/pids` and checked immediately before initialization. It never resumed native inference because all cards remained occupied. There is no established fixed-duration external autokill policy.

A five-minute TT profile measured about 11.56 GiB live allocation per card; stale shared-memory records must not be summed across dead PIDs. Exact compute utilization was unavailable in that firmware. These are TT measurements, not NVIDIA estimates. Slurm's allocated resource isolation replaces manual device-availability polling on HPC; TT-specific leak workarounds must not be represented as CUDA requirements without evidence.

## 10. NVIDIA requirements and acceptance gates

HPC project: `${REPO_ROOT}/a-mem-locomo`; SSH `<hpc-host>`. Use its own virtual environment. Setup, tests, launch, monitoring and analysis run in named tmux sessions with logs and exit records. GPU inference must additionally run in Slurm allocations; never on the login node, and never select an unallocated physical GPU or stop another user's process.

Use the same pinned weights in BF16 CUDA with efficient attention and cache, same prompts, schemas, stochastic policy and memory/scoring semantics. CPU embedding/tokenization/scoring remain permitted; CPU LLM fallback is forbidden. CUDA numeric/attention/cache behavior and the new run seed digests differ from TT and must be disclosed. Validate the full 131,072-token configured capacity; an actual long-context GPU probe is required before claiming it works in this environment. Chunked prefill must preserve all context rather than chunk/summarize memories.

Prefer four one-GPU independent Slurm workers if policy allows, with queued jobs safely running on fewer available devices. Keep original logical shard ownership for clarity. Persist per-turn and per-question state, source/config/environment fingerprints, Slurm IDs, GPU identities and exits. Handle preemption/time limits by durable-boundary stops and documented resume, not blind loops after unknown failures. Full run starts only after:

- Original-source downloads pass hashes and full dataset counts.
- Exact author-prompt extraction and scoring/retrieval parity checks pass.
- Failed QA, overflow, no-leakage, checkpoint/stale-index, retry-cap and frozen-memory tests pass.
- Slurm CUDA forward, constrained JSON, all four stages, restart and allocated-GPU checks pass.
- Long-context capacity and realistic token throughput are measured; estimates state assumptions.
- Final sources/config/environment and a clean initial output directory are frozen in a new manifest.

Completion requires all 5,882 notes, all 1,986 unique QA records, all ten frozen memory/checkpoint completions, six metrics, four plots, traceable failures and an official NeurIPS-format report. Generate the report honestly as partial until those conditions hold. Inspect every PDF page before marking the report final. Never publish automatically.

## 11. Research workflow and open risks

Append concise dated entries to `results/further_work/observations.md`; preserve the user's edits. Separate measured facts, suspected causes and untested experiments. Candidate questions include stale-vector retrieval, duplicate context, temporal change versus correction, invalid/insufficient evidence annotations, and constrained-output behavior in a small model. None is a demonstrated contribution yet.

Main risks: schema budget exhaustion, long-context memory/runtime, queue/preemption, gated model access, growing trace storage, lexical-scoring artifacts, image-dependent questions and partial-run selection bias. Preserve raw evidence rather than change scientific settings mid-run to increase a score. Any scientific change needs a new run ID; operational fixes need versioned rationale and hash evidence.

## References

- [A-Mem author repository](https://github.com/WujiangXu/A-mem/tree/c3bb1ee6d4825b13460dd26be1444b5f870ddb64)
- [LoCoMo author repository](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)
- [Pinned Llama model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/0cb88a4f764b7a12671c53f0838cd831a0843b95)
- [LM Format Enforcer 0.11.3](https://github.com/noamgat/lm-format-enforcer/tree/v0.11.3)
- [Transformers 4.53 Llama implementation/documentation](https://huggingface.co/docs/transformers/v4.53.0/en/model_doc/llama)
- [Official NeurIPS 2026 formatting kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)

### Source provenance correction found during independent download

The original TT README labels the A-Mem repository as `agiresearch/A-mem`, but its pinned commit is not resolvable there. The exact commit resolves at the author repository `WujiangXu/A-mem`. HPC must download from this verified location and compare source/prompt behavior. This corrects provenance metadata, without substituting another algorithm revision.
