# Mem-α → LoCoMo inference-only generalization

This experiment compares the released RL-trained Memalpha-4B writer with frozen base Qwen3-4B. Both use the author's three memory stores, Qwen-Agent tool prompt/parser, BM25 retrieval and the same frozen Qwen3-32B reader. No training, optimizer, reward update, fine-tuning or score-driven prompt selection is enabled.

The target is all 10 released conversations and all 1,986 QA items per arm: 841 single-hop, 282 multi-hop, 321 temporal, 96 open-domain, and 446 adversarial. Outputs comprise 20 frozen memories and 3,972 scheduled QA results. Event summarization and multimodal generation are outside this run.

## Locations and setup

HPC backing root: `${REPO_ROOT}/memalpha-locomo-generalization`.

Requested alias: `${REPO_ROOT}/memalpha-locomo-generalization`.

The source is a complete clone in `upstream/`, pinned to `9c0e1f6482f8ea7a322012a8155247966285946f`. Model IDs/revisions and all inference settings are frozen in `config.json`. The independent `.venv` uses Python 3.12, vLLM 0.8.5, Torch 2.6.0, Transformers 4.51.3, Qwen-Agent 0.0.29, and OpenAI 1.78.1. The resolved dependency list and model SHA/LFS verification receipts are retained under `results/setup/` and `models/`.

`scripts/setup_environment.py` and `scripts/download_models.py` are idempotent asset-preparation entry points intended for durable server tmux sessions. They never modify the environments or model stores of older experiments. Model weights remain frozen BF16. Training dependencies are omitted.

## Protocol

Only explicit conversation fields enter construction: speaker names, original session dates, turn text, and all supplied BLIP captions, including captions with missing image URLs. Sidecars retain source dialogue IDs and exact character coverage. Gold questions, answers, evidence, observations, and supplied summaries are excluded from construction. LoCoMo input SHA-256 is `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

The author's LME preprocessing is adapted to named LoCoMo speakers: split/pack chronologically at 2,048 Qwen3-32B tokens, repeat timestamp headers on split sessions, flush packed chunks at 1,800 tokens, and preserve source text without loss. This yields 138 chunks and 212,567 input tokens; maximum chunk length is 2,040 tokens. Named speakers are not relabeled as user/assistant to activate the author's pair-specific special case.

Each writer sees core plus the latest five semantic and five episodic entries and emits one completion per chunk, with ordered tool execution. Writer temperature is 0, output cap 2,048 and thinking is disabled. Upstream core truncation at 512 tiktoken tokens is preserved. Both writers share tokenizer vocabulary and tool formatting. Invalid model operations are logged as outcomes; no corrective LLM continuation is added.

Reader retrieval uses the raw question, BM25 top 20 per non-core store and all core. The category-specific LoCoMo task instruction is passed separately after the original Mem-α structured-memory system prompt. Retrieval depth is reduced for the upstream 30,000-token system bound and the complete 32,768-token served-context/output-reserve bound. Reader temperature is 0.7, output cap 2,048 and thinking is disabled. Seed 42 is fixed; identical stochastic outputs across hardware/batching are not guaranteed.

The official single-question prompt branches are used verbatim, including category 2's date instruction and category 5's two-option format. Option randomization is precomputed once with `random.Random(42)` in dataset order: abstention is option a for 220 items, b for 226. Both arms reuse the same manifest. Only the question instruction and frozen memory reach the reader; evaluator metadata does not.

## Compatibility changes

- `compatibility.patch` records changes to upstream memory storage: disable unused embedding API calls in BM25-only evaluation, validate storage inputs and guard short-ID collisions. Opaque four-character IDs are derived reproducibly from the seed, memory state and operation content, allowing exact replay of journaled completions.
- `bridge.py` reuses upstream prompt construction, tool parsing, CRUD and BM25/system-prompt code. It calls local vLLM directly instead of the upstream HTTP loop. This removes transport errors and preserves separate independent reader requests, model generation defaults and explicit experiment settings.
- `locomo.py` supplies the missing dataset adapter, lossless chunk boundaries, official task prompts and scoring. Gold metadata remains separate from construction records.
- The stale upstream category-5 interface reads `answer`, absent in 444 cases. This adapter consistently uses `adversarial_answer` as the distractor and never treats it as a correct reference.
- Official legacy answer mapping and category-specific scores are preserved, including known parser weaknesses. Strict option-selection correctness is an additional diagnostic. Four ambiguous annotations are flagged and excluded only from the clean strict diagnostic, never from headline coverage.
- Atomic per-chunk/per-question checkpoints record source/config/input/memory hashes. GPU operation checks and process ownership protect resumable stages. No retrieval/consolidation heuristic is tuned on benchmark scores.

This is **official LoCoMo task prompts and compatible scoring applied to Mem-α memory**. It is not an exact reproduction of LoCoMo's original reader model, raw-context prompt, decoding configuration or broader non-QA tasks.

## Validation and execution

Run the CPU checks in the isolated environment, with the experiment root as working directory:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python run.py prepare --synthetic
.venv/bin/python run.py prepare
```

From the HPC login node, dispatch each writer smoke stage on one eligible GPU. `ops.py dispatch` selects <gpu-node>, then <gpu-node>, checking physical occupancy, scheduler reservations, free memory, and GPU UUID locks. Its output identifies the durable tmux session and status receipt.

```bash
.venv/bin/python ops.py dispatch --name smoke_memalpha --gpu-count 1 --min-host-gib 64 -- .venv/bin/python run.py build --arm memalpha --synthetic
.venv/bin/python ops.py dispatch --name smoke_base --gpu-count 1 --min-host-gib 64 -- .venv/bin/python run.py build --arm qwen3_base --synthetic
```

After both writers complete and release their leases:

```bash
.venv/bin/python ops.py dispatch --name smoke_reader --gpu-count 4 --min-host-gib 128 -- .venv/bin/python run.py answer --synthetic
```

After the reader completes, validate and freeze before any benchmark inference:

```bash
.venv/bin/python run.py validate-smoke
.venv/bin/python run.py score --synthetic
.venv/bin/python run.py freeze
```

Then dispatch the two full writer arms; after both complete, dispatch the full reader:

```bash
.venv/bin/python ops.py dispatch --name full_memalpha --gpu-count 1 --min-host-gib 64 -- .venv/bin/python run.py build --arm memalpha
.venv/bin/python ops.py dispatch --name full_base --gpu-count 1 --min-host-gib 64 -- .venv/bin/python run.py build --arm qwen3_base
.venv/bin/python ops.py dispatch --name full_reader --gpu-count 4 --min-host-gib 128 -- .venv/bin/python run.py answer
```

The reader command must be issued only after all 20 memories are frozen. If `results/setup/reader_runtime.json` records required NCCL compatibility flags, manual reader launches must include those exact environment settings; the durable pipeline applies them automatically. Resource wrappers never submit scheduler jobs or signal other users' processes. A stage stopped due to resource loss can be dispatched again using a fresh operation name; the runner resumes completed immutable checkpoints without duplicate generation. An invalid saved reader response can be retried up to three attempts with identical settings and retained error receipts. Integrity mismatches stop the run rather than overwrite artifacts.

```bash
.venv/bin/python run.py status
.venv/bin/python run.py score
```

Results are under `runs/official_memalpha_locomo_v1/`. The report includes every scheduled question, with missing or failed responses scored zero and coverage shown separately. Category-5 binary accuracy is mixed with category-specific token-F1 in the overall task score; it is not labeled uniform F1. Per-question CSV/JSON, category CSV, paired per-conversation deltas, raw model outputs, retrieval traces, tool failures, compression and runtime measurements accompany the report.

## Durable full-run supervisor

After successful synthetic validation and protocol freeze, the full experiment can be supervised on the login node in tmux:

```bash
tmux new-session -d -s memalpha_full_pipeline -c ${REPO_ROOT}/memalpha-locomo-generalization '.venv/bin/python pipeline.py > results/pipeline/worker.log 2>&1'
```

Create `results/pipeline/` before that command. The supervisor runs both writers, waits for their leases to release, launches the shared reader, and generates the report. It retries resource interruptions with the same frozen configuration; scientific/runtime failures stop for inspection. Its status is `results/pipeline/status.json`. Re-running the supervisor resumes existing operations and validated artifacts. The supervisor's STOP file requests a stop of its own running operations.

The reader's tested communication settings are recorded in `results/setup/reader_runtime.json` and propagated unchanged by the supervisor. The first synthetic reader attempt is retained separately for diagnosis if hardware communication needed adjustment. Such startup attempts do not consume or score benchmark questions.

## Final artifact verification

After scoring completes, verify the full artifact graph independently of the model and scorer:

```bash
.venv/bin/python verification/verify_run.py . --output runs/official_memalpha_locomo_v1/verification/integrity.json
.venv/bin/python verification/package_run.py .
```

This requires exactly 20 frozen memories and 3,972 answer receipts. It checks checkpoint chains, configuration/source/input/memory hashes, the shared seeded question/option manifest, retrieval alignment, and report coverage. It exits nonzero for missing or inconsistent artifacts and never invokes inference or changes scores. The report includes a separate reporting-code hash because presentation changes are outside the scientific source freeze.

Packaging additionally requires a completed supervisor and fully released experiment processes. It produces `runs/official_memalpha_locomo_v1/run_manifest.json`, indexing artifact sizes and SHA-256 checksums plus the model revisions, launch commands, and hardware evidence. Weight files remain on the HPC and are represented by their independently verified model manifest.

Pinned primary sources: [Mem-α implementation](https://github.com/wangyu-ustc/Mem-alpha/tree/9c0e1f6482f8ea7a322012a8155247966285946f), [LoCoMo data and task prompts](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376), and [LoCoMo scoring](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py).
