from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_accumulated_reward(
    frame: np.ndarray,
    accumulated_reward: float,
    *,
    accumulated_reward_terms: dict[str, float] | None = None,
) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    lines = [f"total: {accumulated_reward:.3f}"]
    if accumulated_reward_terms:
        lines.extend(f"{label}: {value:.3f}" for label, value in accumulated_reward_terms.items())

    line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(right - left for left, _top, right, _bottom in line_bboxes)
    line_heights = [bottom - top for _left, top, _right, bottom in line_bboxes]
    line_spacing = 2
    text_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    margin = 8

    x = image.width - text_width - margin
    y = image.height - text_height - margin

    draw.rectangle(
        [(x - 6, y - 4), (x + text_width + 6, y + text_height + 4)],
        fill=(0, 0, 0, 160),
    )
    current_y = y
    for line, line_height in zip(lines, line_heights, strict=True):
        draw.text((x, current_y), line, font=font, fill=(255, 255, 255, 255))
        current_y += line_height + line_spacing
    return np.asarray(image)
