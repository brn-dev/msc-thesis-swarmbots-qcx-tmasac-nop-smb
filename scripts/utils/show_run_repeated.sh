#!/usr/bin/env sh

set -eu

usage() {
    cat <<'EOF'
Usage: show_run_repeated.sh [--count] [PID|--pid PID]

Show active run_repeated.sh process registrations without requesting a stop.
With no arguments, all active repeated runners are shown.
Use --count to print only the active runner count.
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

script_dir=$(
    cd -- "$(dirname -- "$0")"
    pwd
)
repo_root=$(
    cd -- "$script_dir/../.."
    pwd
)
control_dir=${SWARMBOTS_RUN_REPEATED_DIR:-$repo_root/.run/run_repeated}

target_pid=
count_only=0

while [ $# -gt 0 ]; do
    case "$1" in
        --count)
            count_only=1
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

if [ -n "$target_pid" ] && ! is_positive_integer "$target_pid"; then
    printf 'Invalid PID: %s\n' "$target_pid" >&2
    exit 1
fi

if [ "$count_only" -ne 0 ] && [ -n "$target_pid" ]; then
    printf '%s\n' '--count cannot be combined with a PID.' >&2
    exit 1
fi

cleanup_stale_control_files

if [ ! -d "$control_dir" ]; then
    if [ "$count_only" -ne 0 ]; then
        printf '0\n'
        exit 0
    fi
    printf 'No active run_repeated.sh processes found.\n'
    exit 0
fi

if [ -n "$target_pid" ]; then
    control_file=$control_dir/$target_pid.control
    if [ ! -f "$control_file" ]; then
        printf 'No active run_repeated.sh registration for PID %s.\n' "$target_pid" >&2
        exit 1
    fi

    printf 'Active run_repeated.sh process:\n'
    print_control_file "$control_file"
    exit 0
fi

active_count=0
for control_file in "$control_dir"/*.control; do
    if [ ! -e "$control_file" ]; then
        continue
    fi

    pid=$(basename -- "$control_file" .control)
    if is_positive_integer "$pid"; then
        active_count=$((active_count + 1))
    fi
done

if [ "$count_only" -ne 0 ]; then
    printf '%s\n' "$active_count"
    exit 0
fi

if [ "$active_count" -eq 0 ]; then
    printf 'No active run_repeated.sh processes found.\n'
    exit 0
fi

printf 'Active run_repeated.sh processes:\n'
for control_file in "$control_dir"/*.control; do
    if [ ! -e "$control_file" ]; then
        continue
    fi

    pid=$(basename -- "$control_file" .control)
    if is_positive_integer "$pid"; then
        print_control_file "$control_file"
    fi
done
