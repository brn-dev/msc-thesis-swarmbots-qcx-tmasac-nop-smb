from __future__ import annotations

import contextvars
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, TypeVar

from loguru import logger


DISCORD_WEBHOOK_ENV_VAR = "SWARMBOTS_DISCORD_WEBHOOK_URL"
DISCORD_CONTENT_LIMIT = 2000
DISCORD_USER_AGENT = "swarm-bots-training-notifier/0.1"
T = TypeVar("T")

_mjw_warning_notification_lock = threading.Lock()
_mjw_warning_notification_keys: set[tuple[str, str]] = set()
_current_run_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "swarmbots_discord_current_run_name",
    default=None,
)


def run_with_discord_notification(
        *,
        run_name: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        run: Callable[[], T],
) -> T:
    error: BaseException | None = None
    current_run_name_token = _current_run_name.set(run_name)
    try:
        return run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            notify_training_run_finished(
                run_name=run_name,
                run_dir=run_dir,
                total_timesteps=total_timesteps,
                algorithm=algorithm,
                error=error,
            )
        finally:
            _current_run_name.reset(current_run_name_token)


def notify_training_run_finished(
        *,
        run_name: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        error: BaseException | None,
) -> bool:
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV_VAR)
    if not webhook_url:
        return False

    status = _get_run_status(
        current_timesteps=int(algorithm.n_total_timesteps),
        total_timesteps=total_timesteps,
        error=error,
    )
    content = _format_training_run_finished_message(
        run_name=run_name,
        status=status,
        run_dir=run_dir,
        total_timesteps=total_timesteps,
        algorithm=algorithm,
        error=error,
    )
    return send_discord_message(content=content, webhook_url=webhook_url)


def send_discord_message(*, content: str, webhook_url: str) -> bool:
    payload = json.dumps({"content": content[:DISCORD_CONTENT_LIMIT]}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": DISCORD_USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
        message = f"Failed to send Discord notification: HTTP {exc.code} {exc.reason}"
        if response_body:
            message = f"{message}: {response_body[:500]}"
        logger.error(message)
        return False
    except urllib.error.URLError as exc:
        logger.error(f"Failed to send Discord notification: {exc.reason}")
        return False
    except OSError as exc:
        logger.error(f"Failed to send Discord notification: {type(exc).__name__}: {exc}")
        return False

    return True


def notify_mjw_nefc_overflow_once(
        *,
        scenario_name: str,
        num_envs: int,
        nconmax: int,
        njmax: int,
        required_njmax: int,
) -> bool:
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV_VAR)
    if not webhook_url:
        return False

    claimed, run_name = _claim_mjw_warning_notification_once(
        warning_type="nefc_overflow",
        scenario_name=scenario_name,
    )
    if not claimed:
        return False

    lines = [
        f"MJW warning: nefc overflow - please increase njmax to {required_njmax}",
        f"machine: {_get_machine_name()}",
        f"run: {run_name or 'unknown'}",
        f"scenario: {scenario_name}",
        f"current caps: nconmax={nconmax}, njmax={njmax}, num_envs={num_envs}",
    ]
    content = "\n".join(lines)
    return send_discord_message(content=content, webhook_url=webhook_url)


def notify_mjw_simulation_instability_once(
        *,
        scenario_name: str,
        num_envs: int,
        unstable_world_indices: list[int],
) -> bool:
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV_VAR)
    if not webhook_url or not unstable_world_indices:
        return False

    claimed, run_name = _claim_mjw_warning_notification_once(
        warning_type="simulation_instability",
        scenario_name=scenario_name,
    )
    if not claimed:
        return False

    displayed_indices = unstable_world_indices[:20]
    indices_text = ", ".join(str(index) for index in displayed_indices)
    if len(displayed_indices) < len(unstable_world_indices):
        indices_text = f"{indices_text}, ..."
    lines = [
        "MJW warning: simulation instability detected",
        f"machine: {_get_machine_name()}",
        f"run: {run_name or 'unknown'}",
        f"scenario: {scenario_name}",
        f"unstable worlds: {len(unstable_world_indices)} / {num_envs} ({indices_text})",
    ]
    return send_discord_message(content="\n".join(lines), webhook_url=webhook_url)


def _claim_mjw_warning_notification_once(
        *,
        warning_type: str,
        scenario_name: str,
) -> tuple[bool, str | None]:
    run_name = _current_run_name.get()
    run_key = run_name if run_name is not None else f"unscoped:{scenario_name}"
    notification_key = warning_type, run_key
    with _mjw_warning_notification_lock:
        if notification_key in _mjw_warning_notification_keys:
            return False, run_name
        _mjw_warning_notification_keys.add(notification_key)
    return True, run_name


def _get_run_status(
        *,
        current_timesteps: int,
        total_timesteps: int,
        error: BaseException | None,
) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if error is not None:
        return "failed"
    if current_timesteps < total_timesteps:
        return "stopped"
    return "finished"


def _get_machine_name() -> str:
    return socket.gethostname()


def _format_training_run_finished_message(
        *,
        run_name: str,
        status: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        error: BaseException | None,
) -> str:
    lines = [
        f"{run_name} {status}",
        f"machine: {_get_machine_name()}",
        # f"run_dir: {Path(run_dir).as_posix()}",
        f"timesteps: {algorithm.n_total_timesteps:,} / {total_timesteps:,}",
#         f"iterations: {algorithm.n_total_iterations:,}",
#         f"updates: {algorithm.n_total_updates:,}",
    ]
    final_return_ema = getattr(algorithm, "_final_return_ema", None)
    if final_return_ema is not None:
        lines.append(f"final_ep_rew_ema: {final_return_ema:.6g}")

    best_return_ema = getattr(algorithm, "_best_return_ema", None)
    if best_return_ema is not None:
        lines.append(f"best_ep_rew_ema: {best_return_ema:.6g}")

    if error is not None:
        lines.append(f"error: {type(error).__name__}: {error}")

    return "\n".join(lines)
