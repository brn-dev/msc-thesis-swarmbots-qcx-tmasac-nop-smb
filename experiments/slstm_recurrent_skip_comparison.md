# sLSTM-TMASAC recurrent skip comparison

This ablation compares the current thesis sLSTM-TMASAC configuration against the
same configuration with a residual connection around each recurrent module.
The only training/model setting changed is `rmat_temporal_residual`.

| Setting | Thesis baseline | Recurrent skip |
| --- | --- | --- |
| Recurrent residual | Off | On |
| Inter-module MLP | On | On |
| Attention and both MLP residuals | On | On |
| Recurrent input LayerNorm | Off | Off |
| Recurrent output projection | Off | Off |
| Internal sLSTM output normalization | On | On |

Both arms retain the attention-first order, two width-256 encoder layers, two
256 → 512 → 256 actor MLPs per layer, the actor-state-conditioned feed-forward
critic, SMB actions, and critic-sourced NOP. Parameter counts are unchanged.
The training budget remains 100 million environment steps with 1,024 parallel
environments, 32 burn-in steps, and 64 learning steps per replay sequence.

Recurrent input LayerNorm and the output projection remain disabled to isolate
the skip connection. Existing baseline launchers are unchanged.

## Run

From the repository root, use the existing baseline launcher and the new launcher
for the same scenario. Each invocation creates a separate run ID. Repeat both
arms equally to compare variability across runs; these launchers do not add a
paired-seed protocol. Training requires the usual MJWarp/CUDA environment.

Find-opening:

```powershell
.\.venv\Scripts\python.exe experiments/thesis_mjw_find_opening/scripts/run_tmasac_slstm.py
.\.venv\Scripts\python.exe experiments/thesis_mjw_find_opening/scripts/run_tmasac_slstm_recurrent_skip.py
```

PO-wall medium:

```powershell
.\.venv\Scripts\python.exe experiments/thesis_mjw_po_wall_medium/scripts/run_tmasac_slstm.py
.\.venv\Scripts\python.exe experiments/thesis_mjw_po_wall_medium/scripts/run_tmasac_slstm_recurrent_skip.py
```

The new runs are saved under
`runs/<thesis suite>/slstm_two_small_actor_state_critic_recurrent_skip/<run ID>`;
baseline runs keep their existing group directory.

## Compare

Run the usual scenario plotter after both groups have results:

```powershell
.\.venv\Scripts\python.exe experiments/thesis_mjw_find_opening/plot_results.py
.\.venv\Scripts\python.exe experiments/thesis_mjw_po_wall_medium/plot_results.py
```

Each writes a separate pair comparison under
`experiments/<thesis suite>/results/recurrent_skip/slstm_tmasac/`, using the
existing 100M-step cutoff and baseline run sources. The ablation is excluded from
the main algorithm comparison. The usual `summarize_results.py` entrypoints also
include the new group in their metrics.
