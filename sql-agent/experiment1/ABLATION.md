# Controller ablation

## Protocol
Same model, SQLite DB, grader (`eval_metrics.py`), and revise budgets.

| Arm | Behavior |
|-----|----------|
| full | Controller ↔ DB talk then SQL ReAct (current `run.py`) |
| no_controller | Skip talk; DB agent receives the raw question as the brief and searches immediately |

## Flag
```bash
python run.py --no-controller --one-per-template   # after wiring
```

Until the matched re-run lands, `metrics/controller_ablation.json` records status=`harness_ready_awaiting_matched_rerun` and cost proxies from the full-system run.
