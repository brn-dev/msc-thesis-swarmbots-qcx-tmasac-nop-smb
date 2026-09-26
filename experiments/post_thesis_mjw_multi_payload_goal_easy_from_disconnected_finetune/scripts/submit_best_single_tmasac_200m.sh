#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/data/fs201347/db72480/swarm-bots"
RUN_SCRIPT="$REPO_ROOT/experiments/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/scripts/continue_single_tmasac_200m.py"
SOURCE_GROUP_DIR="$REPO_ROOT/runs/post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune/tmasac_from_disconnected_finetune_single"
SUBMITTER="$REPO_ROOT/scripts/utils/submit_checkpoint_transfer_pairs.sh"
EXPECTED_RUN_COUNT=5
SELECTED_RUN_COUNT=4

DRY_RUN_ARG=()
case "${1:-}" in
    --dry-run)
        DRY_RUN_ARG=(--dry-run)
        shift
        ;;
    --help|-h)
        echo "Usage: bash $0 [--dry-run]"
        echo
        echo "Ranks the five completed single-payload runs at their 100M"
        echo "evaluation endpoint, drops the worst, and continues four models"
        echo "for another 200M transitions."
        exit 0
        ;;
    *)
        ;;
esac

if [[ $# -gt 0 ]]; then
    echo "ERROR: Unknown option: $1" >&2
    exit 2
fi

find_initial_final_checkpoint() {
    local run_dir="$1"
    local selected_checkpoint=""
    local selected_steps=-1
    local checkpoint_path

    shopt -s nullglob
    for checkpoint_path in "$run_dir"/models/*_final.pt; do
        local checkpoint_name="${checkpoint_path##*/}"
        if [[ "$checkpoint_name" =~ ^model_([0-9]+)_steps_final\.pt$ ]]; then
            local steps=$((10#${BASH_REMATCH[1]}))
            if (( selected_steps < 0 || steps < selected_steps )); then
                selected_steps=$steps
                selected_checkpoint="$checkpoint_path"
            fi
        fi
    done
    shopt -u nullglob

    if [[ -z "$selected_checkpoint" ]]; then
        return 1
    fi
    printf '%s\n' "$selected_checkpoint"
}

read_100m_endpoint_metrics() {
    local eval_log="$1"
    awk -F';' '
        NR == 1 {
            for (column = 1; column <= NF; column++) {
                if ($column == "eval_milestone_pct") milestone_column = column
                if ($column == "eval_deterministic_success_rate") success_column = column
                if ($column == "eval_deterministic_ep_rew__mean") reward_column = column
                if ($column == "timesteps") timesteps_column = column
            }
            next
        }
        milestone_column && success_column && reward_column && timesteps_column &&
        $milestone_column + 0 == 100 {
            printf "%s\t%s\t%s\n", $success_column, $reward_column, $timesteps_column
            found = 1
            exit
        }
        END {
            if (!found) exit 1
        }
    ' "$eval_log"
}

collect_ranked_candidates() {
    local run_dir
    shopt -s nullglob
    for run_dir in "$SOURCE_GROUP_DIR"/*; do
        [[ -d "$run_dir" ]] || continue

        local checkpoint_path
        if ! checkpoint_path=$(find_initial_final_checkpoint "$run_dir"); then
            echo "ERROR: Missing final checkpoint below $run_dir" >&2
            return 1
        fi

        local eval_log="$run_dir/eval_log.csv"
        if [[ ! -f "$eval_log" ]]; then
            echo "ERROR: Missing evaluation log: $eval_log" >&2
            return 1
        fi

        local metrics
        if ! metrics=$(read_100m_endpoint_metrics "$eval_log"); then
            echo "ERROR: Missing 100% evaluation row in $eval_log" >&2
            return 1
        fi

        local success_rate
        local episode_reward
        local timesteps
        IFS=$'\t' read -r success_rate episode_reward timesteps <<< "$metrics"
        printf '%s\t%s\t%s\t%s\n' \
            "$success_rate" "$episode_reward" "$timesteps" "$checkpoint_path"
    done
    shopt -u nullglob
}

ranked_output=$(collect_ranked_candidates | sort -t $'\t' -k1,1gr -k2,2gr -k3,3gr)
if [[ -z "$ranked_output" ]]; then
    echo "ERROR: No completed single-payload runs found below $SOURCE_GROUP_DIR" >&2
    exit 1
fi
mapfile -t RANKED_ROWS <<< "$ranked_output"

if [[ ${#RANKED_ROWS[@]} -ne $EXPECTED_RUN_COUNT ]]; then
    echo "ERROR: Expected $EXPECTED_RUN_COUNT completed runs, found ${#RANKED_ROWS[@]}." >&2
    exit 1
fi

echo "Single-payload 100M ranking"
echo "Primary metric: deterministic success rate; tie-breaker: deterministic reward"

SUBMITTER_MODEL_ARGS=()
for rank_idx in "${!RANKED_ROWS[@]}"; do
    IFS=$'\t' read -r success_rate episode_reward timesteps checkpoint_path \
        <<< "${RANKED_ROWS[$rank_idx]}"
    if (( rank_idx < SELECTED_RUN_COUNT )); then
        selection="selected"
        SUBMITTER_MODEL_ARGS+=(--model "$checkpoint_path")
    else
        selection="ignored"
    fi
    printf '%d. success=%s reward=%s steps=%s [%s]\n   %s\n' \
        "$((rank_idx + 1))" "$success_rate" "$episode_reward" "$timesteps" \
        "$selection" "$checkpoint_path"
done
echo

exec bash "$SUBMITTER" \
    --job-name "single-payload-continue-200m" \
    --run-script "$RUN_SCRIPT" \
    --source-group-dir "$SOURCE_GROUP_DIR" \
    "${SUBMITTER_MODEL_ARGS[@]}" \
    "${DRY_RUN_ARG[@]}"
