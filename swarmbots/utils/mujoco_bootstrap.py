from __future__ import annotations

import os
import sys


def configure_mujoco_gl_backend() -> None:
    if sys.platform != "linux":
        return
    if os.environ.get("MUJOCO_GL"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    os.environ["MUJOCO_GL"] = "egl"

