# PO-wall disconnected-start fine-tuning

This experiment continues a trained PO-wall TMASAC baseline for 20M additional
environment transitions. Each episode starts from a compact pre-connected layout
with all connection edges removed; normal reset settling is still applied. Training
uses the original 50-layout pool (seeds 42000--42049), while the baked-in frozen
evaluation hook uses a disjoint 50-layout pool (seeds 3000000--3000049).

Run one continuation for each source checkpoint:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac.py <checkpoint.pt>
```

The hook evaluates after +10M and +20M transitions (50% and 100% of the
fine-tuning window) and writes `eval_log.csv` in the continuation run directory.
Evaluation and live video recording are disabled so headless cluster runs do not
require an EGL display; this does not disable numerical evaluation.

Separate 50M launchers evaluate after +25M and +50M transitions and write under
`runs/thesis_mjw_po_wall_disconnected_finetune_50m/`:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac_50m.py <checkpoint.pt>
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac_no_connectors_50m.py <checkpoint.pt>
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_mat_qcx_50m.py <checkpoint.pt>
```

For the matched harder-wall experiment, the following launchers retain the same
disconnected training and evaluation pools but increase only the wall height from
0.3 to the established hard-wall value of 0.4. They write to
`runs/thesis_mjw_po_wall_disconnected_finetune_hard_wall_50m/`:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac_hard_wall_50m.py <checkpoint.pt>
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\run_tmasac_no_connectors_hard_wall_50m.py <checkpoint.pt>
```

Both continue for 50M transitions and evaluate numerically after +25M and +50M.
The actuator-only control is required to establish whether independent modules
can learn to cross the taller wall; wall height alone does not guarantee this.

Evaluate the final/best checkpoints from all four TMASAC groups (medium/hard wall
and connector-enabled/actuator-only) on disjoint, fully disconnected 4- and
5-unit pools with one command:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\evaluate_tmasac_50m.py
```

The evaluator defaults to 512 deterministic episodes per unit count and writes
one independently resumable JSON/CSV pair per case beneath
`experiments/thesis_mjw_po_wall_disconnected_finetune/results/evaluation_50m/`.
Connection counts are summarized across all episodes and separately across
successful and unsuccessful episodes. Outcome-specific fields are empty when a
run has no episodes with that outcome.
Use `--case medium_connectors` (repeatable) to select a subset, and `--resume`
or `--overwrite` for existing outputs.

After evaluation, `global_summary.json` is written beside the four case files.
It pools metrics across checkpoints with exact observation counts, retains
per-unit-count results, reports checkpoint-level variability, measures how often
successful and unsuccessful episodes contain never-connected units, and provides
connector/no-connector and hard/medium effect differences. Rebuild it from
existing case files without running the policies again with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\evaluate_tmasac_50m.py --summary-only
```

Generate hard-wall plots and a separate hard-wall summary with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\plot_hard_wall_results.py
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\summarize_hard_wall_results.py
```

These write only to
`experiments/thesis_mjw_po_wall_disconnected_finetune/hard_wall_results/`.
Record the hard-wall connector-enabled and actuator-only checkpoints with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\record_tmasac_hard_wall_50m.py
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\record_tmasac_no_connectors_hard_wall_50m.py
```

The recorders use the disjoint evaluation pool, rebuild the environment with
wall height 0.4, and write beneath the hard-wall run group's `recordings/`
directory without reading or replacing the medium-wall recordings.

The no-connectors TMASAC variant exposes only actuator actions to the policy and
always sends `-1` for every physical connector. It restores the complete actuator
head and removes only the connector columns from critic/NOP action-input weights.
The old connector head is discarded and optimizer state starts fresh.

The MAT-QCX launcher expects the standard thesis MAT-QCX configuration: PPO with
1024 environments, four rollout steps per environment, sign-magnitude Beta actions,
and NOP enabled.

Generate grouped and per-run training curves for all three 50M variants with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\plot_results.py
```

Training plots are written to
`experiments/thesis_mjw_po_wall_disconnected_finetune/results/`. Generate the
final training-metric summary with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\summarize_results.py
```

The summary is written to
`experiments/thesis_mjw_po_wall_disconnected_finetune/results/final_metrics.json`.
Record the normal and actuator-only TMASAC checkpoints respectively with:

```powershell
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\record_tmasac_50m.py
.venv\Scripts\python.exe experiments\thesis_mjw_po_wall_disconnected_finetune\scripts\record_tmasac_no_connectors_50m.py
```

Each recorder automatically selects the final checkpoint from every run in its
corresponding 50M group, falling back to the latest best checkpoint for incomplete
runs. Both default to the disjoint evaluation seed range and record 4- and 5-unit
pools. Command-line options are forwarded to the shared recorder; for example,
append `--episodes 5 --cuda-idx 1`.

The zero-fine-tuning baseline is the existing disconnected-pool evaluation:

```powershell
.venv\Scripts\python.exe experiments\evaluate_thesis_mjw_disconnected_pools.py --target po_wall_tmasac --unit-counts 4 5
```

For the connection-use ablation, evaluate a fine-tuned checkpoint with connector
actions forced to disconnect:

```powershell
.venv\Scripts\python.exe experiments\evaluate_thesis_mjw_disconnected_pools_no_connections.py --target po_wall_tmasac --unit-counts 4 5 --po-wall-checkpoint <checkpoint.pt>
```
