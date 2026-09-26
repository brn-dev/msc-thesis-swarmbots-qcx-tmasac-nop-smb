#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/data/fs201347/db72480/swarm-bots"
RUN_SCRIPT="$REPO_ROOT/experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/run_tmasac.py"
SOURCE_GROUP_DIR="$REPO_ROOT/runs/thesis_mjw_po_wall_disconnected_finetune_50m/tmasac_baseline"
SUBMITTER="$REPO_ROOT/scripts/utils/submit_checkpoint_transfer_pairs.sh"

TASK_VARIANT="easy"
SUBMITTER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --variant requires easy, very_easy, or single." >&2
                exit 2
            fi
            TASK_VARIANT="$2"
            shift 2
            ;;
        --dry-run)
            SUBMITTER_ARGS+=(--dry-run)
            shift
            ;;
        --help|-h)
            echo "Usage: bash $0 [--dry-run] [--variant easy|very_easy|single]"
            echo
            echo "Discovers all source checkpoints and submits one single-GPU"
            echo "Slurm job per pair of models."
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

case "$TASK_VARIANT" in
    easy|very_easy|single)
        ;;
    *)
        echo "ERROR: Unknown variant: $TASK_VARIANT" >&2
        exit 2
        ;;
esac

exec bash "$SUBMITTER" \
    --job-name "multi-payload-${TASK_VARIANT//_/-}" \
    --run-script "$RUN_SCRIPT" \
    --source-group-dir "$SOURCE_GROUP_DIR" \
    --run-arg=--variant \
    --run-arg="$TASK_VARIANT" \
    "${SUBMITTER_ARGS[@]}"
