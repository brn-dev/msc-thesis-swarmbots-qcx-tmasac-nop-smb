# PO wall medium multi-sample experiments

This suite compares the SAC multi-sample action-loss settings for three
continuous action distributions:

- sign-magnitude Beta (Gumbel straight-through),
- RQS with 6 bins, and
- Bernstein degree 8.

Every run uses 4 target samples. The paired launchers use either 6 or 8 actor
samples, with stratified quantile sampling enabled for distributions that
support it. Run a launcher from this directory's `scripts` folder, for example:

```powershell
.\.venv\Scripts\python.exe experiments/post_thesis_mjw_po_wall_medium_action_sampling/scripts/run_tmasac_rqs_6_actor8_target4.py
```

Use `plot_results.py` and `summarize_results.py` after runs have produced logs.
Both scripts also load the established TMASAC baseline from the thesis PO-Wall
Medium run sources (actor=1, target=1) for direct comparison.
