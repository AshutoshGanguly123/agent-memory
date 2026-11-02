# Mem-α → LoCoMo inference-only evaluation

Official LoCoMo category-specific task prompts and compatible scoring applied to Mem-alpha structured memory.

Coverage: 3972/3972 completed answers; 20/20 frozen memories.

| Category | Questions per arm | Mem-α | Base Qwen3-4B | Difference |
|---|---:|---:|---:|---:|
| multi_hop | 282 | 0.1691 | 0.2055 | -0.0363 |
| temporal | 321 | 0.1811 | 0.2319 | -0.0508 |
| open_domain | 96 | 0.1045 | 0.1147 | -0.0102 |
| single_hop | 841 | 0.1993 | 0.2012 | -0.0019 |
| adversarial | 446 | 0.8139 | 0.8027 | +0.0112 |
| Overall micro | — | 0.3255 | 0.3376 | -0.0121 |
| Category macro | — | 0.2936 | 0.3112 | -0.0176 |
| Nonadversarial micro | — | 0.1841 | 0.2030 | -0.0189 |
| Exact unambiguous adversarial choice accuracy | — | 0.7713 | 0.7601 | +0.0112 |
| Exact unambiguous choice accuracy excluding four anomalies | — | 0.7715 | 0.7602 | +0.0113 |

## Paired conversation scores

Each row uses every scheduled question in that conversation, including missing/error outputs at zero.

| Conversation | Questions per arm | Completed Mem-α / base | Mem-α | Base Qwen3-4B | Difference |
|---|---:|---:|---:|---:|---:|
| conv-26 | 199 | 199 / 199 | 0.3342 | 0.3445 | -0.0103 |
| conv-30 | 105 | 105 / 105 | 0.3692 | 0.4007 | -0.0315 |
| conv-41 | 193 | 193 / 193 | 0.3584 | 0.3689 | -0.0105 |
| conv-42 | 260 | 260 / 260 | 0.3251 | 0.3354 | -0.0102 |
| conv-43 | 242 | 242 / 242 | 0.3117 | 0.3652 | -0.0534 |
| conv-44 | 158 | 158 / 158 | 0.3527 | 0.3389 | +0.0138 |
| conv-47 | 190 | 190 / 190 | 0.3042 | 0.3109 | -0.0067 |
| conv-48 | 239 | 239 / 239 | 0.3237 | 0.2544 | +0.0693 |
| conv-49 | 196 | 196 / 196 | 0.2901 | 0.3333 | -0.0431 |
| conv-50 | 204 | 204 / 204 | 0.3151 | 0.3648 | -0.0497 |

## Interpretation and protocol

The comparison changes only the frozen memory-writing checkpoint. Both arms use identical source chunks, memory tools, BM25 retrieval, category-specific task prompts, adversarial option order, and frozen Qwen3-32B reader settings. No parameters are trained or selected on LoCoMo.

Scores use the official category-specific token-F1 rules and legacy adversarial parser/scorer. Adversarial scores are binary, so the overall number is a mixed task score, not a uniform token-F1 metric. Missing/error outputs score zero and remain in the denominator.

The additional exact unambiguous choice diagnostic accepts an explicit a/b label, a label followed by its matching exact option text, or a response equal to one distinct option text. It normalizes case, whitespace and terminal sentence punctuation. Conflicting label/text and unresolved or missing responses score zero. This reporting-only diagnostic does not alter the official headline score or legacy prediction mapping.

The reader uses Mem-α’s structured-memory system prompt, temperature 0.7, 2,048 output tokens and seed 42. LoCoMo’s original reader generation used different settings. This is a memory-transfer evaluation with official task prompts, not an exact end-to-end reproduction of the original LoCoMo models. One seed per arm does not quantify run-to-run sampling variability.

The adversarial distractor comes from adversarial_answer, fixing the stale missing-answer interface. conv-26 questions 167/178 contain answer=No and distractor=Yes; conv-30 questions 79/103 have a distractor reading Not mentioned. All four stay in the headline score; the additional clean strict diagnostic excludes them. Question indices are zero-based. Invalid short responses accepted by the legacy parser are flagged in per_question.json.

Memory construction uses only dialogue text, dates, speakers and supplied image captions. Dialogue IDs remain in provenance sidecars. No gold evidence, questions, supplied summaries or observations enter construction. QA cannot update frozen memory. Retrieval evidence recall is not reported because generated memory entries do not provide reliable exact source attribution.

## Memory size and compression

| Arm | Measured / frozen / scheduled memories | Matched source tokens | Stored tokens total | Mean stored tokens | Token reduction |
|---|---:|---:|---:|---:|---:|
| memalpha | 10 / 10 / 10 | 212,567 | 90,703 | 9,070.3 | 57.3% |
| qwen3_base | 10 / 10 / 10 | 212,567 | 55,427 | 5,542.7 | 73.9% |

Memory sizes count stored core, semantic and episodic content with the reader tokenizer, excluding entry IDs and prompt wrappers. Reduction is one minus total stored tokens divided by source-chunk tokens for the same measured conversations. Missing measurements remain unavailable.

## Model tokens and generation latency

| Arm | Phase | Calls / retries | Model input tokens | Model output tokens | Allocated batch generation (s) | Mean batch latency (s) | Timed calls |
|---|---|---:|---:|---:|---:|---:|---:|
| memalpha | writer | 138 / — | 561,720 | 238,930 | 1,006.81 | 58.17 | 138 / 138 |
| memalpha | reader | 1986 / 0 | 9,619,863 | 33,575 | 1,786.44 | 28.78 | 1986 / 1986 |
| memalpha | **Total** | — | **10,181,583** | **272,505** | — | — | — |
| qwen3_base | writer | 138 / — | 510,872 | 95,197 | 698.79 | 43.20 | 138 / 138 |
| qwen3_base | reader | 1986 / 0 | 7,855,700 | 32,043 | 1,536.70 | 24.70 | 1986 / 1986 |
| qwen3_base | **Total** | — | **8,366,572** | **127,240** | — | — | — |
| **Both arms** | **Total** | — | **18,548,155** | **399,745** | — | — | — |

Model input totals include repeated context and instructions. Reader usage includes unique archived retries plus the latest attempt; only the latest receipt is scored. Mean batch latency is the observed generation-call wall time for each timed request, excluding model loading, prompt preparation and tool execution. Allocated batch generation divides each batch's wall time evenly among its requests; it is not individual request latency. Gaps in archived attempts are flagged as reader_usage_incomplete_questions.

## Latest phase GPU allocation

| Phase | Status | Elapsed wall time (s) | Visible GPUs | Allocated GPU seconds | Allocated GPU hours |
|---|---|---:|---:|---:|---:|
| writer_memalpha | complete | 1,110.53 | 1 | 1,110.53 | 0.3085 |
| writer_qwen3_base | complete | 755.96 | 1 | 755.96 | 0.2100 |
| reader_shared | complete | 3,520.20 | 4 | 14,080.80 | 3.9113 |

Latest recorded phase invocation elapsed wall time multiplied by visible GPU count, including model loading. Not measured GPU compute time or cumulative lease time across restarts; the reader row is shared by both arms. Complete lease and hardware receipts are under [results/operations](../../../results/operations/).

Per-store memory sizes, tool-error counts, truncation counts, detailed usage and source status paths are retained in metrics.json.

## Artifacts and reproducibility

Reporting code SHA-256: `19b440e73517d07af6e5ed896e11cfb3447144ba423633fd3db6dfb24b4257df`. This identifies report.py separately from the immutable scientific source freeze.

- metrics.json: aggregates, paired differences, per-conversation results, and operational measurements.
- per_question.csv / per_question.json: all scheduled questions including failures.
- per_category.csv: comparison table.
- per_conversation.csv: paired scores and completion coverage for every conversation.
- annotation_anomalies.json: explicitly flagged annotation cases.
- ../freeze.json: frozen source, configuration and prepared-input hashes.
- ../<arm>/conversations: raw writer prompts/completions, tool traces, atomic checkpoints, and frozen memory.
- ../<arm>/answers: raw reader prompts/completions and retrieval traces.

See the experiment README.md for restart commands.

Reader communication/runtime settings: [results/setup/reader_runtime.json](../../../results/setup/reader_runtime.json), SHA-256 `dc01787ad73d6506d3daa82da31eb79b112c0fd6f5185f293135992f7fc4ada1`. The saved environment is included in metrics.json provenance.
