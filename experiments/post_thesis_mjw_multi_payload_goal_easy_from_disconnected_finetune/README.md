# Easy multi-payload transfer from disconnected-swarm fine-tuning

This post-thesis experiment transfers the final connector-enabled TMASAC models
from `thesis_mjw_po_wall_disconnected_finetune_50m` to the easy multi-payload
goal and trains each model for 100M additional environment transitions.

The target uses the established easy configuration: at most two payloads, with
one or two active payloads sampled equally, and continuous connector actions.
Two easier variants are also available:

- `very_easy`: at most two payloads, with a 90% chance of one active payload.
- `single`: exactly one payload.

Because the source and target observation spaces differ, transfer loading keeps
all task-independent policy tensors with matching names and shapes, freshly
initializes every global-observation encoder and its input normalization, and
does not restore PO-wall normalization statistics or optimizer state. The
source timestep counter is retained, so evaluation milestones and the stopping
point are relative to the 100M transfer window.

Run a single source checkpoint (recommended for one cluster job per model):

```powershell
.venv\Scripts\python.exe experiments\post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune\scripts\run_tmasac.py <checkpoint.pt>
```

When all source runs are available on the MUSICA cluster, use the self-submitting
Slurm wrapper. It discovers every final checkpoint and submits `ceil(N / 2)`
independent jobs. Each job requests one GPU and starts up to two training
processes on it:

```powershell
bash experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/submit_all_tmasac.sh
```

Select the target variant with `--variant`; `easy` remains the default:

```powershell
bash experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/submit_all_tmasac.sh --variant very_easy
bash experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/submit_all_tmasac.sh --variant single
```

The wrapper must be invoked with `bash`, not `sbatch`, because it groups the
models before submitting the jobs. To inspect the discovered checkpoints and
their job assignments without submitting or training, use:

```powershell
bash experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/submit_all_tmasac.sh --variant very_easy --dry-run
```

Each allocated job writes training output to
`$DATA/slurm_logs/<job-id>/stream1.log` and, for two-model jobs,
`stream2.log`. The top-level Slurm output records the model-to-stream mapping.

Outputs are written under
`runs/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/` and
reuse each source run ID.

## Continue the four best single-payload runs

After all five `single` runs finish their initial 100M transfer window, rank
them by deterministic success rate at the 100% evaluation milestone. Ties are
broken by deterministic mean episode reward. The following wrapper drops the
lowest-ranked run and strictly resumes the other four, including their
normalization and optimizer state, for another 200M transitions:

```powershell
bash experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/submit_best_single_tmasac_200m.sh
```

Inspect the complete ranking and the two resulting GPU-job assignments without
submitting anything with:

```powershell
bash experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/submit_best_single_tmasac_200m.sh --dry-run
```

The continuation outputs are written under
`runs/post_thesis_mjw_multi_payload_goal_single_continue_200m/`.
