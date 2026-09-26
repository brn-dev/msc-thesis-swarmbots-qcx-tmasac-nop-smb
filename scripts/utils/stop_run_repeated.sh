#!/usr/bin/env sh

set -eu

usage() {
    cat <<'EOF'
Usage: stop_run_repeated.sh [PID|--pid PID|--all]

Request active run_repeated.sh process(es) to stop after their current run.

With one active repeated runner, no arguments are needed. With multiple active
runners, pass a PID shown by this command or use --all.
EOF
}

is_positive_integer() {
    case "$1" in
        ''|*[!0-9]*)
            return 1
            ;;
        *)
            [ "$1" -gt 0 ]
            ;;
    esac
}

cleanup_stale_control_files() {
    if [ ! -d "$control_dir" ]; then
        return 0
    fi

    for control_file in "$control_dir"/*.control; do
        if [ ! -e "$control_file" ]; then
            continue
        fi

        pid=$(basename -- "$control_file" .control)
        if ! is_positive_integer "$pid" || ! kill -0 "$pid" 2>/dev/null; then
            rm -f -- "$control_file"
        fi
    done
}

print_control_file() {
    control_file=$1

    pid=$(basename -- "$control_file" .control)
    stop_file=$(sed -n '1p' "$control_file")
    training_script=$(sed -n '2p' "$control_file")
    runs=$(sed -n '3p' "$control_file")
    started_at=$(sed -n '4p' "$control_file")
    completed_runs=$(sed -n '5p' "$control_file")
    stop_requested=no
    if [ -n "$stop_file" ] && [ -f "$stop_file" ]; then
        stop_requested=yes
    fi

    printf '  %s' "$pid"
    if [ -n "$started_at" ]; then
        printf ' started=%s' "$started_at"
    fi
    if [ -n "$runs" ]; then
        printf ' runs=%s' "$runs"
        if [ -n "$completed_runs" ]; then
            printf ' completed=%s/%s' "$completed_runs" "$runs"
        else
            printf ' completed=?/%s' "$runs"
        fi
    fi
    printf ' stop_requested=%s' "$stop_requested"
    if [ -n "$training_script" ]; then
        printf ' script=%s' "$training_script"
    fi
    printf '\n'
}

request_stop_for_pid() {
    target_pid=$1
    control_file=$control_dir/$target_pid.control

    if [ ! -f "$control_file" ]; then
        printf 'No active run_repeated.sh registration for PID %s.\n' "$target_pid" >&2
        exit 1
    fi

    stop_file=$(sed -n '1p' "$control_file")
    training_script=$(sed -n '2p' "$control_file")

    if [ -z "$stop_file" ]; then
        printf 'Invalid run_repeated.sh registration for PID %s.\n' "$target_pid" >&2
        exit 1
    fi

    mkdir -p -- "$(dirname -- "$stop_file")"
    : > "$stop_file"
    printf 'Stop requested for PID %s' "$target_pid"
    if [ -n "$training_script" ]; then
        printf ' (%s)' "$training_script"
    fi
    printf '.\n'
}

script_dir=$(
    cd -- "$(dirname -- "$0")"
    pwd
)
repo_root=$(
    cd -- "$script_dir/../.."
    pwd
)
control_dir=${SWARMBOTS_RUN_REPEATED_DIR:-$repo_root/.run/run_repeated}

stop_all=0
target_pid=

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            stop_all=1
            shift
            ;;
        --pid)
            if [ $# -lt 2 ]; then
                printf 'Missing value for --pid\n' >&2
                exit 1
            fi
            target_pid=$2
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            if [ -n "$target_pid" ]; then
                printf 'Unexpected argument: %s\n' "$1" >&2
                usage >&2
                exit 1
            fi
            target_pid=$1
            shift
            ;;
    esac
done

if [ "$stop_all" -ne 0 ] && [ -n "$target_pid" ]; then
    printf 'Use either --all or a PID, not both.\n' >&2
    exit 1
fi

if [ -n "$target_pid" ] && ! is_positive_integer "$target_pid"; then
    printf 'Invalid PID: %s\n' "$target_pid" >&2
    exit 1
fi

cleanup_stale_control_files

if [ ! -d "$control_dir" ]; then
    printf 'No active run_repeated.sh processes found.\n' >&2
    exit 1
fi

if [ -n "$target_pid" ]; then
    request_stop_for_pid "$target_pid"
    exit 0
fi

active_count=0
only_pid=

for control_file in "$control_dir"/*.control; do
    if [ ! -e "$control_file" ]; then
        continue
    fi

    pid=$(basename -- "$control_file" .control)
    if ! is_positive_integer "$pid"; then
        continue
    fi

    active_count=$((active_count + 1))
    only_pid=$pid
done

if [ "$active_count" -eq 0 ]; then
    printf 'No active run_repeated.sh processes found.\n' >&2
    exit 1
fi

if [ "$stop_all" -ne 0 ]; then
    for control_file in "$control_dir"/*.control; do
        if [ ! -e "$control_file" ]; then
            continue
        fi

        pid=$(basename -- "$control_file" .control)
        if is_positive_integer "$pid"; then
            request_stop_for_pid "$pid"
        fi
    done
    exit 0
fi

if [ "$active_count" -eq 1 ]; then
    request_stop_for_pid "$only_pid"
    exit 0
fi

printf 'Multiple active run_repeated.sh processes found. Pick one PID or use --all:\n' >&2
for control_file in "$control_dir"/*.control; do
    if [ ! -e "$control_file" ]; then
        continue
    fi

    pid=$(basename -- "$control_file" .control)
    if ! is_positive_integer "$pid"; then
        continue
    fi

    print_control_file "$control_file" >&2
done
exit 1
