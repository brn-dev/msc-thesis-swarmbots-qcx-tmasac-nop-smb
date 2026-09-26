#!/usr/bin/env bash

#SBATCH --account=p201347
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=3-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/fs201347/db72480/slurm_logs/%x-%j.out
#SBATCH --error=/data/fs201347/db72480/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/data/fs201347/db72480/swarm-bots"
PYTHON="$REPO_ROOT/.venv/bin/python"
SUBMIT_SCRIPT="$REPO_ROOT/scripts/utils/submit_checkpoint_transfer_pairs.sh"
START_DELAY=10

DRY_RUN=false
EXECUTE=false
JOB_NAME=""
RUN_SCRIPT=""
SOURCE_GROUP_DIR=""
RUN_ARGS=()
REQUESTED_MODELS=()
MODELS=()

usage() {
    echo "Usage: bash $0 --job-name NAME --run-script PATH --source-group-dir PATH [options]"
    echo
    echo "Options:"
    echo "  --run-arg ARG  Forward ARG to the training script; may be repeated."
    echo "  --model PATH   Submit this checkpoint instead of discovering all; may be repeated."
    echo "  --dry-run      Show checkpoint-to-job assignments without submitting."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --execute)
            EXECUTE=true
            shift
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --run-script)
            RUN_SCRIPT="$2"
            shift 2
            ;;
        --source-group-dir)
            SOURCE_GROUP_DIR="$2"
            shift 2
            ;;
        --run-arg)
            RUN_ARGS+=("$2")
            shift 2
            ;;
        --run-arg=*)
            RUN_ARGS+=("${1#*=}")
            shift
            ;;
        --model)
            REQUESTED_MODELS+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "ERROR: Unknown option: $1" >&2
            exit 2
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$JOB_NAME" || -z "$RUN_SCRIPT" || -z "$SOURCE_GROUP_DIR" ]]; then
    echo "ERROR: --job-name, --run-script, and --source-group-dir are required." >&2
    exit 2
fi

discover_models() {
    MODELS=()
    shopt -s nullglob

    local run_dir
    for run_dir in "$SOURCE_GROUP_DIR"/*; do
        [[ -d "$run_dir" ]] || continue

        local latest_checkpoint=""
        local latest_steps=-1
        local checkpoint_path
        for checkpoint_path in "$run_dir"/models/*_final.pt; do
            local checkpoint_name="${checkpoint_path##*/}"
            if [[ "$checkpoint_name" =~ ^model_([0-9]+)_steps_final\.pt$ ]]; then
                local steps=$((10#${BASH_REMATCH[1]}))
                if (( steps > latest_steps )); then
                    latest_steps=$steps
                    latest_checkpoint="$checkpoint_path"
                fi
            fi
        done

        if [[ -n "$latest_checkpoint" ]]; then
            MODELS+=("$latest_checkpoint")
        fi
    done

    shopt -u nullglob
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo "ERROR: No final checkpoints found below $SOURCE_GROUP_DIR" >&2
        exit 1
    fi
}

select_models() {
    if [[ ${#REQUESTED_MODELS[@]} -gt 0 ]]; then
        MODELS=("${REQUESTED_MODELS[@]}")
    else
        discover_models
    fi
}

print_plan() {
    local job_count=$(( (${#MODELS[@]} + 1) / 2 ))
    echo "Job name: $JOB_NAME"
    echo "${#MODELS[@]} model(s), $job_count Slurm job(s), 1 GPU per job"

    local model_idx
    for model_idx in "${!MODELS[@]}"; do
        echo "Job $((model_idx / 2 + 1)): ${MODELS[$model_idx]}"
    done
}

if [[ "$DRY_RUN" == true ]]; then
    if [[ "$EXECUTE" == true || ${#MODELS[@]} -gt 0 ]]; then
        echo "ERROR: --dry-run cannot be combined with worker arguments." >&2
        exit 2
    fi
    select_models
    print_plan
    exit 0
fi

if [[ "$EXECUTE" == false ]]; then
    if [[ ${#MODELS[@]} -gt 0 ]]; then
        echo "ERROR: Checkpoint paths are reserved for allocated worker jobs." >&2
        exit 2
    fi

    select_models
    print_plan
    echo

    for ((model_idx = 0; model_idx < ${#MODELS[@]}; model_idx += 2)); do
        job_models=("${MODELS[$model_idx]}")
        if (( model_idx + 1 < ${#MODELS[@]} )); then
            job_models+=("${MODELS[$((model_idx + 1))]}")
        fi

        worker_args=(
            --execute
            --job-name "$JOB_NAME"
            --run-script "$RUN_SCRIPT"
            --source-group-dir "$SOURCE_GROUP_DIR"
        )
        for run_arg in "${RUN_ARGS[@]}"; do
            worker_args+=(--run-arg "$run_arg")
        done
        worker_args+=("${job_models[@]}")

        sbatch --job-name="$JOB_NAME" --gres="gpu:1" "$SUBMIT_SCRIPT" "${worker_args[@]}"
    done
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: --execute is reserved for the allocated Slurm job." >&2
    exit 1
fi
if [[ ${#REQUESTED_MODELS[@]} -gt 0 ]]; then
    echo "ERROR: --model is only valid when submitting jobs." >&2
    exit 2
fi
if [[ ${#MODELS[@]} -lt 1 || ${#MODELS[@]} -gt 2 ]]; then
    echo "ERROR: An allocated worker job requires one or two checkpoint paths." >&2
    exit 2
fi

module load Python/3.13.5-GCCcore-14.3.0
cd "$REPO_ROOT"

if [[ -f "$HOME/.swarmbots_env" ]]; then
    source "$HOME/.swarmbots_env"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DATA_ROOT="${DATA:-/data/fs201347/db72480}"
LOG_DIR="$DATA_ROOT/slurm_logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $RUN_SCRIPT" >&2
    exit 1
fi

echo "============================================================"
echo "$JOB_NAME job"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Stream logs:  $LOG_DIR"
printf 'Model:        %s\n' "${MODELS[@]}"
echo "============================================================"

nvidia-smi -L
echo

PIDS=()
for model_idx in "${!MODELS[@]}"; do
    stream_number=$((model_idx + 1))
    log_file="$LOG_DIR/stream${stream_number}.log"
    echo "Starting stream $stream_number"
    echo "  model: ${MODELS[$model_idx]}"
    echo "  log:   $log_file"
    "$PYTHON" "$RUN_SCRIPT" \
        "${MODELS[$model_idx]}" \
        "${RUN_ARGS[@]}" \
        --cuda-idx 0 \
        >"$log_file" 2>&1 &
    PIDS+=("$!")

    if (( model_idx < ${#MODELS[@]} - 1 )); then
        sleep "$START_DELAY"
    fi
done

EXIT_CODE=0
for model_idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$model_idx]}"; then
        echo "Completed stream $((model_idx + 1)): ${MODELS[$model_idx]}"
    else
        return_code=$?
        echo "ERROR: ${MODELS[$model_idx]} failed with exit code $return_code" >&2
        EXIT_CODE=1
    fi
done

exit "$EXIT_CODE"
