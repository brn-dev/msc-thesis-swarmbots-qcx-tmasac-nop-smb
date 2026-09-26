#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/data/fs201347/db72480/swarm-bots"
RUN_SCRIPT="$REPO_ROOT/experiments/post_thesis_mjw_bridge_static_from_disconnected_finetune/scripts/run_tmasac.py"
SOURCE_GROUP_DIR="$REPO_ROOT/runs/thesis_mjw_po_wall_disconnected_finetune_50m/tmasac_baseline"
SUBMITTER="$REPO_ROOT/scripts/utils/submit_checkpoint_transfer_pairs.sh"

case "${1:-}" in
    --dry-run)
        DRY_RUN_ARG=(--dry-run)
        shift
        ;;
    --help|-h)
        echo "Usage: bash $0 [--dry-run]"
        echo
        echo "Discovers all source checkpoints and submits one single-GPU"
        echo "Slurm job per pair of models."
        exit 0
        ;;
    *)
        DRY_RUN_ARG=()
        ;;
esac

if [[ $# -gt 0 ]]; then
    echo "ERROR: Unknown option: $1" >&2
    exit 2
fi

exec bash "$SUBMITTER" \
    --job-name "bridge-static-transfer" \
    --run-script "$RUN_SCRIPT" \
    --source-group-dir "$SOURCE_GROUP_DIR" \
    "${DRY_RUN_ARG[@]}"
