#!/usr/bin/env sh

set -eu

usage() {
    cat <<'EOF'
Usage: wait_then_run_repeated.sh [GATE...] SCRIPT_PATH [RUNS] [OPTIONS] [-- SCRIPT_ARGS...]

Wait until all selected gates have been satisfied for a stable period, then
call scripts/utils/run_repeated.sh.

Gates:
  VRAM amount         Required free VRAM in MiB. Plain numbers are MiB.
                      Supported suffixes: M, MiB, G, GiB, T, TiB.
                      Examples: 24000, 24G.
  RUN COUNT           Maximum number of other active repeated runs. Use the
                      suffix R, for example 2R.

Arguments:
  SCRIPT_PATH         Training script path passed to run_repeated.sh.
  RUNS                Number of runs passed to run_repeated.sh. Defaults to 5.

Options:
  --gpu INDEX             GPU index for nvidia-smi. Defaults to 0.
  --stable-for SECONDS    Required continuous availability. Defaults to 300.
  --check-interval SECONDS
                          Seconds between VRAM checks. Defaults to 30.
  --python PYTHON         Python executable passed to run_repeated.sh.
                          Defaults to "python".
  --delay SECONDS         Delay between repeated runs. Defaults to 0.
  --stop-file PATH        Stop-request file passed to run_repeated.sh.
  --lock-dir PATH         Lock directory for serializing the VRAM wait/start
                          decision. Defaults to a per-GPU directory under
                          ${TMPDIR:-/tmp}.
  --no-lock               Disable wait/start locking.
  --help                  Show this help text.

If SWARMBOTS_DISCORD_WEBHOOK_URL is set, a Discord notification is sent when
the wait ends and the repeated runs are about to start.

Gate arguments must come before SCRIPT_PATH. You can provide one gate or both.

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

is_positive_integer() {
    is_non_negative_integer "$1" && [ "$1" -gt 0 ]
}

try_parse_memory_mib() {
    memory_text=$1

    case "$memory_text" in
        *[Tt][Ii][Bb])
            memory_number=${memory_text%[Tt][Ii][Bb]}
            memory_multiplier=1048576
            ;;
        *[Tt][Bb])
            memory_number=${memory_text%[Tt][Bb]}
            memory_multiplier=1048576
            ;;
        *[Tt])
            memory_number=${memory_text%[Tt]}
            memory_multiplier=1048576
            ;;
        *[Gg][Ii][Bb])
            memory_number=${memory_text%[Gg][Ii][Bb]}
            memory_multiplier=1024
            ;;
        *[Gg][Bb])
            memory_number=${memory_text%[Gg][Bb]}
            memory_multiplier=1024
            ;;
        *[Gg])
            memory_number=${memory_text%[Gg]}
            memory_multiplier=1024
            ;;
        *[Mm][Ii][Bb])
            memory_number=${memory_text%[Mm][Ii][Bb]}
            memory_multiplier=1
            ;;
        *[Mm][Bb])
            memory_number=${memory_text%[Mm][Bb]}
            memory_multiplier=1
            ;;
        *[Mm])
            memory_number=${memory_text%[Mm]}
            memory_multiplier=1
            ;;
        *)
            memory_number=$memory_text
            memory_multiplier=1
            ;;
    esac

    if ! is_positive_integer "$memory_number"; then
        return 1
    fi

    printf '%s\n' "$((memory_number * memory_multiplier))"
}

parse_memory_mib() {
    if ! memory_mib=$(try_parse_memory_mib "$1"); then
        printf 'Invalid VRAM amount: %s\n' "$1" >&2
        exit 1
    fi

    printf '%s\n' "$memory_mib"
}

try_parse_run_limit() {
    run_limit_text=$1

    case "$run_limit_text" in
        *[Rr])
            run_limit_text=${run_limit_text%[Rr]}
            ;;
        *)
            return 1
            ;;
    esac

    if ! is_non_negative_integer "$run_limit_text"; then
        return 1
    fi

    printf '%s\n' "$run_limit_text"
}

query_free_vram_mib() {
    query_gpu_index=$1
    free_vram=$(
        nvidia-smi \
            --id="$query_gpu_index" \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null \
            | tr -d '[:space:]'
    )

    if ! is_non_negative_integer "$free_vram"; then
        printf 'Could not read free VRAM for GPU %s with nvidia-smi.\n' "$query_gpu_index" >&2
        exit 1
    fi

    printf '%s\n' "$free_vram"
}

count_active_run_repeated_processes() {
    if [ ! -f "$show_run_repeated_script" ]; then
        printf 'show_run_repeated.sh not found next to this script: %s\n' "$show_run_repeated_script" >&2
        exit 1
    fi

    if ! active_count=$(sh "$show_run_repeated_script" --count); then
        printf 'Could not read active run count from show_run_repeated.sh.\n' >&2
        exit 1
    fi
    if ! is_non_negative_integer "$active_count"; then
        printf 'Could not read active run count from show_run_repeated.sh.\n' >&2
        exit 1
    fi

    printf '%s\n' "$active_count"
}

cleanup_lock() {
    if [ "${lock_acquired:-0}" -eq 1 ]; then
        rm -f -- "$lock_dir/pid"
        rmdir -- "$lock_dir" 2>/dev/null || true
        lock_acquired=0
    fi
}

acquire_lock() {
    while ! mkdir -- "$lock_dir" 2>/dev/null; do
        lock_pid=
        if [ -f "$lock_dir/pid" ]; then
            lock_pid=$(cat -- "$lock_dir/pid" 2>/dev/null || true)
        fi

        if is_positive_integer "${lock_pid:-}" && ! kill -0 "$lock_pid" 2>/dev/null; then
            printf 'Removing stale lock: %s\n' "$lock_dir"
            rm -f -- "$lock_dir/pid"
            rmdir -- "$lock_dir" 2>/dev/null || true
            continue
        fi

        if [ -n "${lock_pid:-}" ]; then
            printf 'Waiting for GPU %s lock held by PID %s: %s\n' "$gpu_index" "$lock_pid" "$lock_dir"
        else
            printf 'Waiting for GPU %s lock: %s\n' "$gpu_index" "$lock_dir"
        fi
        sleep "$check_interval_seconds"
    done

    lock_acquired=1
    printf '%s\n' "$$" > "$lock_dir/pid"
}

send_start_notification() {
    if [ -z "${SWARMBOTS_DISCORD_WEBHOOK_URL:-}" ]; then
        return 0
    fi

    machine_name=$(hostname 2>/dev/null || uname -n)
    notification_message=$(
        cat <<EOF
wait_then_run_repeated starting
machine: $machine_name
gpu: $gpu_index
script: $script_path
runs: $runs
EOF
    )

    if ! "$python_executable" "$repo_root/scripts/utils/send_discord_notification.py" --message "$notification_message" >/dev/null; then
        printf 'Failed to send Discord start notification.\n' >&2
    fi
}

describe_wait_targets() {
    if [ -n "$required_free_vram_mib" ] && [ -n "$max_other_repeated_runs" ]; then
        printf 'Waiting for GPU %s to have at least %s MiB free and at most %s other repeated runs for %s seconds.\n' \
            "$gpu_index" "$required_free_vram_mib" "$max_other_repeated_runs" "$stable_for_seconds"
    elif [ -n "$required_free_vram_mib" ]; then
        printf 'Waiting for GPU %s to have at least %s MiB free for %s seconds.\n' \
            "$gpu_index" "$required_free_vram_mib" "$stable_for_seconds"
    else
        printf 'Waiting for at most %s other repeated runs for %s seconds.\n' \
            "$max_other_repeated_runs" "$stable_for_seconds"
    fi
}

wait_conditions_met() {
    wait_status_message=

    if [ -n "$required_free_vram_mib" ]; then
        free_vram_mib=$(query_free_vram_mib "$gpu_index")
        if [ "$free_vram_mib" -lt "$required_free_vram_mib" ]; then
            wait_status_message="GPU $gpu_index free VRAM is $free_vram_mib MiB; waiting for $required_free_vram_mib MiB."
            return 1
        fi
    fi

    if [ -n "$max_other_repeated_runs" ]; then
        active_run_count=$(count_active_run_repeated_processes)
        if [ "$active_run_count" -gt "$max_other_repeated_runs" ]; then
            wait_status_message="There are $active_run_count active repeated runs; waiting for at most $max_other_repeated_runs other repeated runs."
            return 1
        fi
    fi

    return 0
}

if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

required_free_vram_mib=
max_other_repeated_runs=
script_path=

while [ $# -gt 0 ]; do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
    esac

    if [ -z "$script_path" ]; then
        if [ -z "$required_free_vram_mib" ] && memory_mib=$(try_parse_memory_mib "$1"); then
            required_free_vram_mib=$memory_mib
            shift
            continue
        fi

        if [ -z "$max_other_repeated_runs" ] && run_limit=$(try_parse_run_limit "$1"); then
            max_other_repeated_runs=$run_limit
            shift
            continue
        fi

        script_path=$1
        shift
        break
    fi

    break
done

if [ -z "$script_path" ]; then
    printf 'Missing SCRIPT_PATH.\n' >&2
    usage >&2
    exit 1
fi

if [ -z "$required_free_vram_mib" ] && [ -z "$max_other_repeated_runs" ]; then
    printf 'Missing gate argument. Provide a VRAM amount, a run limit like 2R, or both.\n' >&2
    usage >&2
    exit 1
fi

runs=5
if [ $# -gt 0 ] && is_non_negative_integer "$1"; then
    runs=$1
    shift
fi

gpu_index=0
stable_for_seconds=300
check_interval_seconds=30
python_executable=python
delay_seconds=0
stop_file=
use_lock=1
lock_dir=

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --gpu\n' >&2
                exit 1
            fi
            if ! is_non_negative_integer "$2"; then
                printf 'Invalid GPU index: %s\n' "$2" >&2
                exit 1
            fi
            gpu_index=$2
            shift 2
            ;;
        --stable-for)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --stable-for\n' >&2
                exit 1
            fi
            if ! is_non_negative_integer "$2"; then
                printf 'Invalid stable duration: %s\n' "$2" >&2
                exit 1
            fi
            stable_for_seconds=$2
            shift 2
            ;;
        --check-interval)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --check-interval\n' >&2
                exit 1
            fi
            if ! is_positive_integer "$2"; then
                printf 'Invalid check interval: %s\n' "$2" >&2
                exit 1
            fi
            check_interval_seconds=$2
            shift 2
            ;;
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
        --lock-dir)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --lock-dir\n' >&2
                exit 1
            fi
            lock_dir=$2
            shift 2
            ;;
        --no-lock)
            use_lock=0
            shift
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
run_repeated_script=$script_dir/run_repeated.sh
show_run_repeated_script=$script_dir/show_run_repeated.sh

if [ ! -f "$run_repeated_script" ]; then
    printf 'run_repeated.sh not found next to this script: %s\n' "$run_repeated_script" >&2
    exit 1
fi

if [ ! -f "$show_run_repeated_script" ]; then
    printf 'show_run_repeated.sh not found next to this script: %s\n' "$show_run_repeated_script" >&2
    exit 1
fi

if [ -z "$lock_dir" ]; then
    lock_dir=${TMPDIR:-/tmp}/swarmbots_wait_then_run_repeated_gpu_${gpu_index}.lock
fi

lock_acquired=0
if [ "$use_lock" -ne 0 ]; then
    trap cleanup_lock EXIT
    trap 'cleanup_lock; exit 130' INT
    trap 'cleanup_lock; exit 143' TERM
    acquire_lock
fi

stable_started_at=

describe_wait_targets

while :; do
    now=$(date '+%s')

    if wait_conditions_met; then
        if [ -z "$stable_started_at" ]; then
            stable_started_at=$now
            if [ -n "$required_free_vram_mib" ] && [ -n "$max_other_repeated_runs" ]; then
                printf 'GPU %s free VRAM is now %s MiB and there are %s active repeated runs; starting stable timer.\n' \
                    "$gpu_index" "$free_vram_mib" "$active_run_count"
            elif [ -n "$required_free_vram_mib" ]; then
                printf 'GPU %s free VRAM is now %s MiB; starting stable timer.\n' "$gpu_index" "$free_vram_mib"
            else
                printf 'There are now %s active repeated runs; starting stable timer.\n' "$active_run_count"
            fi
        fi

        stable_elapsed=$((now - stable_started_at))
        if [ "$stable_elapsed" -ge "$stable_for_seconds" ]; then
            if [ -n "$required_free_vram_mib" ] && [ -n "$max_other_repeated_runs" ]; then
                printf 'GPU %s stayed above %s MiB and there were at most %s other repeated runs for %s seconds. Starting repeated run.\n' \
                    "$gpu_index" "$required_free_vram_mib" "$max_other_repeated_runs" "$stable_elapsed"
            elif [ -n "$required_free_vram_mib" ]; then
                printf 'GPU %s stayed above %s MiB for %s seconds. Starting repeated run.\n' \
                    "$gpu_index" "$required_free_vram_mib" "$stable_elapsed"
            else
                printf 'There were at most %s other repeated runs for %s seconds. Starting repeated run.\n' \
                    "$max_other_repeated_runs" "$stable_elapsed"
            fi
            break
        fi
    else
        if [ -n "$stable_started_at" ]; then
            printf '%s\n' "$wait_status_message"
            printf 'Resetting stable timer.\n'
        else
            printf '%s\n' "$wait_status_message"
        fi
        stable_started_at=
    fi

    sleep "$check_interval_seconds"
done

cleanup_lock
send_start_notification

if [ -n "$stop_file" ]; then
    sh "$run_repeated_script" "$script_path" "$runs" \
        --python "$python_executable" \
        --delay "$delay_seconds" \
        --stop-file "$stop_file" \
        -- "$@"
else
    sh "$run_repeated_script" "$script_path" "$runs" \
        --python "$python_executable" \
        --delay "$delay_seconds" \
        -- "$@"
fi
