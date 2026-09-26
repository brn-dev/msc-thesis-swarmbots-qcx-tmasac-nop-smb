from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmbots.learn.discord_notifications import DISCORD_WEBHOOK_ENV_VAR, send_discord_message, _get_machine_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a test Discord notification.")
    parser.add_argument(
        "--webhook-url",
        default=os.environ.get(DISCORD_WEBHOOK_ENV_VAR),
        help=f"Discord webhook URL. Defaults to ${DISCORD_WEBHOOK_ENV_VAR}.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Optional message body. A timestamped swarm-bots test message is used by default.",
    )
    return parser.parse_args()


def default_message() -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    return "\n".join([
        "swarm-bots Discord notification test",
        f"machine: {_get_machine_name()}",
        f"time: {timestamp}",
    ])


def main() -> int:
    args = parse_args()
    webhook_url = args.webhook_url
    if not webhook_url:
        print(f"Set {DISCORD_WEBHOOK_ENV_VAR} or pass --webhook-url.", file=sys.stderr)
        return 2

    message = args.message if args.message is not None else default_message()
    if not send_discord_message(content=message, webhook_url=webhook_url):
        return 1

    print("Discord test notification sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
