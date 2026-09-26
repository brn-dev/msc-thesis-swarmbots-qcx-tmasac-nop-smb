#!/usr/bin/env sh

set -eu

usage() {
    cat <<'EOF'
Usage: run_repeated.sh SCRIPT_PATH [RUNS] [--python PYTHON] [--delay SECONDS] [--stop-file PATH] [-- SCRIPT_ARGS...]

Arguments:
  SCRIPT_PATH         Training script path. Relative paths are resolved against:
                      current directory, repo root, and the scripts directory.
  RUNS                Number of runs. Defaults to 5.

Options:
  --python PYTHON     Python executable to use. Defaults to "python".
  --delay SECONDS     Delay between runs. Defaults to 0.
  --stop-file PATH    Stop-request file. Defaults to a managed per-process
                      file used by scripts/utils/stop_run_repeated.sh.
  --help              Show this help text.

Any arguments after `--` are forwarded to the training script.
EOF
}

is_non_negative_integer() {
    case "$1" in
        ''|*[!0-9]*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

abspath() {
    target_path=$1
    target_dir=$(dirname -- "$target_path")
    target_name=$(basename -- "$target_path")
    (
        cd -- "$target_dir"
        printf '%s/%s\n' "$(pwd)" "$target_name"
    )
}

resolve_training_script_path() {
    requested_path=$1
    shift

    case "$requested_path" in
        /*|[A-Za-z]:/*)
            if [ ! -f "$requested_path" ]; then
                printf 'Training script not found: %s\n' "$requested_path" >&2
                exit 1
            fi
            abspath "$requested_path"
            return 0
            ;;
    esac

    for base_directory in "$@"; do
        candidate_path=$base_directory/$requested_path
        if [ -f "$candidate_path" ]; then
            abspath "$candidate_path"
            return 0
        fi
    done

    printf 'Training script not found: %s\n' "$requested_path" >&2
    exit 1
}

cleanup_control_file() {
    if [ -n "${active_control_file:-}" ]; then
        rm -f -- "$active_control_file" "$active_control_file.tmp"
    fi
    if [ -n "${stop_file:-}" ]; then
        rm -f -- "$stop_file"
    fi
}

write_control_file() {
    {
        printf '%s\n' "$stop_file"
        printf '%s\n' "$training_script"
        printf '%s\n' "$runs"
        printf '%s\n' "$runner_started_at"
        printf '%s\n' "$completed_runs"
    } > "$active_control_file.tmp"
    mv -f -- "$active_control_file.tmp" "$active_control_file"
}

send_completion_notification() {
    if [ -z "${SWARMBOTS_DISCORD_WEBHOOK_URL:-}" ]; then
        return 0
    fi

    machine_name=$(hostname 2>/dev/null || uname -n)

    if [ "$has_failures" -ne 0 ]; then
        notification_status=failed
        notification_summary="Some repeated runs failed."
    elif [ "$stop_requested" -ne 0 ]; then
        notification_status=stopped
        notification_summary="Repeated runs stopped before finishing every requested run."
    else
        notification_status=finished
        notification_summary="All repeated runs finished successfully."
    fi

    notification_message=$(
        cat <<EOF
run_repeated $notification_status
machine: $machine_name
script: $training_script
runs: $runs
completed_runs: $completed_runs
$notification_summary
EOF
    )

    if [ "$has_failures" -ne 0 ]; then
        notification_message=$(printf '%s\nfailed runs:\n%s' "$notification_message" "$failed_runs")
    fi

    if ! "$python_executable" "$repo_root/scripts/utils/send_discord_notification.py" --message "$notification_message" >/dev/null; then
        printf 'Failed to send Discord completion notification.\n' >&2
    fi
}

if [ $# -eq 0 ]; then
    usage >&2
    exit 1
fi

script_path=
runs=5
python_executable=python
delay_seconds=0
stop_file=

if [ "$1" = "--help" ]; then
    usage
    exit 0
fi

script_path=$1
shift

if [ $# -gt 0 ] && is_non_negative_integer "$1"; then
    runs=$1
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --python\n' >&2
                exit 1
            fi
            python_executable=$2
            shift 2
            ;;
        --delay)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --delay\n' >&2
                exit 1
            fi
            if ! is_non_negative_integer "$2"; then
                printf 'Invalid delay value: %s\n' "$2" >&2
                exit 1
            fi
            delay_seconds=$2
            shift 2
            ;;
        --stop-file)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --stop-file\n' >&2
                exit 1
            fi
            stop_file=$2
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if ! is_non_negative_integer "$runs"; then
    printf 'Invalid run count: %s\n' "$runs" >&2
    exit 1
fi

script_dir=$(
    cd -- "$(dirname -- "$0")"
    pwd
)
repo_root=$(
    cd -- "$script_dir/../.."
    pwd
)
scripts_root=$repo_root/scripts
current_dir=$(pwd)
training_script=$(resolve_training_script_path "$script_path" "$current_dir" "$repo_root" "$scripts_root" "$script_dir")
control_dir=${SWARMBOTS_RUN_REPEATED_DIR:-$repo_root/.run/run_repeated}
if [ -z "$stop_file" ]; then
    stop_file=$control_dir/$$.stop
fi
stop_file_dir=$(dirname -- "$stop_file")
mkdir -p -- "$control_dir" "$stop_file_dir"
rm -f -- "$stop_file"
active_control_file=$control_dir/$$.control
original_pythonpath=${PYTHONPATH-}
failed_runs=
has_failures=0
stop_requested=0
completed_runs=0
run_index=1
runner_started_at=$(date '+%Y-%m-%dT%H:%M:%S')

case "$(uname -s 2>/dev/null || printf unknown)" in
    CYGWIN*|MINGW*|MSYS*)
        pythonpath_separator=';'
        ;;
    *)
        pythonpath_separator=':'
        ;;
esac

write_control_file
trap cleanup_control_file EXIT
trap 'cleanup_control_file; exit 130' INT
trap 'cleanup_control_file; exit 143' TERM

printf 'To stop after the current run finishes, run: %s/stop_run_repeated.sh\n\n' "$script_dir"

while [ "$run_index" -le "$runs" ]; do
    started_at=$(date '+%Y-%m-%dT%H:%M:%S')
    printf '[%s/%s] Starting %s at %s\n' "$run_index" "$runs" "$training_script" "$started_at"

    if [ -n "$original_pythonpath" ]; then
        pythonpath_value=$repo_root$pythonpath_separator$original_pythonpath
    else
        pythonpath_value=$repo_root
    fi

    if (
        export PYTHONPATH=$pythonpath_value
        cd -- "$scripts_root"
        "$python_executable" "$training_script" "$@"
    ); then
        exit_code=0
    else
        exit_code=$?
    fi

    finished_at=$(date '+%Y-%m-%dT%H:%M:%S')
    if [ "$exit_code" -eq 0 ]; then
        printf '[%s/%s] Finished successfully at %s\n' "$run_index" "$runs" "$finished_at"
    else
        has_failures=1
        failed_runs="${failed_runs}${run_index}\t${exit_code}\t${started_at}\t${finished_at}
"
        printf '[%s/%s] Failed with exit code %s at %s\n' "$run_index" "$runs" "$exit_code" "$finished_at" >&2
    fi

    completed_runs=$run_index
    write_control_file

    if [ -f "$stop_file" ]; then
        stop_requested=1
        printf 'Stop requested via %s; not starting further runs.\n' "$stop_file"
        break
    fi

    if [ "$run_index" -lt "$runs" ] && [ "$delay_seconds" -gt 0 ]; then
        sleep "$delay_seconds"
    fi

    run_index=$((run_index + 1))
done

if [ "$has_failures" -ne 0 ]; then
    if [ "$stop_requested" -ne 0 ]; then
        printf '\nStopped after %s of %s requested runs.\n' "$completed_runs" "$runs"
    fi
    printf '\nFailed runs:\n'
    printf 'Run\tExitCode\tStartedAt\tFinishedAt\n'
    printf '%s' "$failed_runs"
    send_completion_notification
    exit 1
fi

if [ "$stop_requested" -ne 0 ]; then
    printf '\nStopped after %s of %s requested runs. Completed runs finished successfully.\n' "$completed_runs" "$runs"
    send_completion_notification
    exit 0
fi

printf '\nAll %s runs finished successfully.\n' "$runs"
send_completion_notification
exit 0
