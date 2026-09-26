DEFAULT_RECORDING_WIDTH = 1280
DEFAULT_RECORDING_HEIGHT = 720


def ensure_mujoco_offscreen_framebuffer(model: object, *, width: int, height: int) -> None:
    global_visual = model.vis.global_
    global_visual.offwidth = max(int(global_visual.offwidth), int(width))
    global_visual.offheight = max(int(global_visual.offheight), int(height))
