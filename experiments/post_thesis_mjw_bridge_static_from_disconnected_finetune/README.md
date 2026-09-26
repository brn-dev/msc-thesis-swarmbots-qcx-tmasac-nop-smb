# Static bridge transfer from disconnected-swarm fine-tuning

This post-thesis experiment transfers the final connector-enabled TMASAC models
from `thesis_mjw_po_wall_disconnected_finetune_50m` to the established static
bridge scenario and trains each model for 100M additional environment
transitions.

Because the source and target observation spaces differ, transfer loading keeps
all task-independent policy tensors with matching names and shapes, freshly
initializes every global-observation encoder and its input normalization, and
does not restore PO-wall normalization statistics or optimizer state. The
source timestep counter is retained, so evaluation milestones and the stopping
point are relative to the 100M transfer window.

Run a single source checkpoint with:

```powershell
.venv\Scripts\python.exe experiments\post_thesis_mjw_bridge_static_from_disconnected_finetune\scripts\run_tmasac.py <checkpoint.pt>
```

On the MUSICA cluster, the Bash wrapper discovers all final source checkpoints
and submits `ceil(N / 2)` jobs. Each job requests one GPU and runs up to two
models in parallel:

```powershell
bash experiments/post_thesis_mjw_bridge_static_from_disconnected_finetune/scripts/submit_all_tmasac.sh
```

Preview the assignments without importing PyTorch or submitting jobs with:

```powershell
bash experiments/post_thesis_mjw_bridge_static_from_disconnected_finetune/scripts/submit_all_tmasac.sh --dry-run
```

Each allocated job writes training output to
`$DATA/slurm_logs/<job-id>/stream1.log` and, for two-model jobs,
`stream2.log`. Outputs are written under
`runs/post_thesis_mjw_bridge_static_from_disconnected_finetune/` and reuse each
source run ID.
