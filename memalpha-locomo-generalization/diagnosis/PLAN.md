# Mem-α negative-result diagnosis

## Decisive question
Why does Mem-α lose on overall micro-F1 vs base Qwen3-4B under our frozen LoCoMo protocol?

## Protocol (locked)
1. **Budget-matched**: cap stored tokens / reader context identically across writers.
2. **Stage labels** on a stratified error sample: writer omit vs retrieval miss vs reader fail.
3. **Reference systems**: no-memory, BM25 chunk retrieval, full-context (if fits).
4. **One intervention** proposed from labels; validate on untouched conversations.

## Current evidence
See `snapshot.json` (copied from official report metrics) and repo-level README headlines.
Official report remains under `../runs/official_memalpha_locomo_v1/report/`.

## Status
Framework + snapshot committed. GPU re-runs for budget-matched arms are next when capacity frees after graph compute-matched launches.
