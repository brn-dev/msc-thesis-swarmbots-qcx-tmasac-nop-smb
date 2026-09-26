from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import moviepy.video.io.ImageSequenceClip
import mujoco
import numpy as np
from loguru import logger

from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRecordingCameraConfig
from swarmbots.utils.recording_overlay import draw_accumulated_reward
from swarmbots.utils.recording_resolution import ensure_mujoco_offscreen_framebuffer


@dataclass(slots=True)
class MJWWorldSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    eq_active: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    time: float


@dataclass(slots=True)
class MJWRecordingConfig:
    video_folder: Path
    video_name_prefix: str
    num_episodes: int
    max_parallel_episodes: int
    fps: int
    fps_mode: str
    frame_stride: int
    width: int
    height: int
    camera: int | str


@dataclass(slots=True)
class _EpisodeSlot:
    world_idx: int
    episode_idx: int
    frames: list[np.ndarray] = field(default_factory=list)
    accumulated_reward: float = 0.0
    accumulated_reward_terms: dict[str, float] = field(default_factory=dict)
    step_count: int = 0
    unstable: bool = False


@dataclass(slots=True)
class _RenderContext:
    data: mujoco.MjData
    renderer: mujoco.Renderer
    camera: int | str | mujoco.MjvCamera


def _write_video_file(*, frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path.as_posix(), logger=None)


class MJWLiveEpisodeRecorder:
    def __init__(self, *, scenario: BaseMJWScenario) -> None:
        self._scenario = scenario
        self._model = scenario.build_model()
        self._config: MJWRecordingConfig | None = None
        self._render_context: _RenderContext | None = None
        self._active_slots_by_world: dict[int, _EpisodeSlot] = {}
        self._episodes_started = 0
        self._episodes_completed = 0
        self._writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mjw-video-write")
        self._writer_futures: list[Future[None]] = []

    def start(
        self,
        *,
        config: MJWRecordingConfig,
        episode_start_world_idx: np.ndarray,
        snapshots_by_world: dict[int, MJWWorldSnapshot],
    ) -> None:
        if self.is_active():
            raise RuntimeError("MJW live recording is already active")

        if config.num_episodes <= 0:
            raise ValueError(f"Expected num_episodes > 0, got {config.num_episodes}")
        if config.max_parallel_episodes <= 0:
            raise ValueError(f"Expected max_parallel_episodes > 0, got {config.max_parallel_episodes}")
        if config.frame_stride <= 0:
            raise ValueError(f"Expected frame_stride > 0, got {config.frame_stride}")
        if config.fps <= 0:
            raise ValueError(f"Expected fps > 0, got {config.fps}")
        if config.fps_mode not in {"compensate_stride", "fixed"}:
            raise ValueError(f"Unsupported fps_mode={config.fps_mode!r}")
        if config.width <= 0 or config.height <= 0:
            raise ValueError(f"Expected positive width/height, got {config.width=} {config.height=}")

        config.video_folder.mkdir(parents=True, exist_ok=True)
        self._cleanup_writer_futures()
        self._close_render_context()

        ensure_mujoco_offscreen_framebuffer(self._model, width=config.width, height=config.height)
        render_context = _RenderContext(
            data=mujoco.MjData(self._model),
            renderer=mujoco.Renderer(self._model, height=config.height, width=config.width),
            camera=self._build_camera(config.camera),
        )

        self._episodes_started = 0
        self._episodes_completed = 0
        self._active_slots_by_world.clear()
        self._config = config
        self._render_context = render_context

        self.on_episode_starts(
            world_idx=episode_start_world_idx,
            snapshots_by_world=snapshots_by_world,
        )

    def is_active(self) -> bool:
        return self._config is not None

    def active_world_indices(self) -> np.ndarray:
        return np.asarray(sorted(self._active_slots_by_world.keys()), dtype=np.int64)

    def get_status(self) -> dict[str, Any]:
        self._cleanup_writer_futures()

        if self._config is None:
            return {
                "active": False,
                "episodes_started": int(self._episodes_started),
                "episodes_completed": int(self._episodes_completed),
                "active_worlds": [],
                "writer_jobs_pending": int(len(self._writer_futures)),
            }

        active_slots = sorted(self._active_slots_by_world.values(), key=lambda slot: slot.episode_idx)
        return {
            "active": True,
            "video_folder": self._config.video_folder.as_posix(),
            "video_name_prefix": self._config.video_name_prefix,
            "num_episodes": int(self._config.num_episodes),
            "max_parallel_episodes": int(self._config.max_parallel_episodes),
            "fps": int(self._config.fps),
            "fps_mode": self._config.fps_mode,
            "effective_fps": int(self._effective_fps(self._config)),
            "frame_stride": int(self._config.frame_stride),
            "width": int(self._config.width),
            "height": int(self._config.height),
            "camera": self._config.camera,
            "episodes_started": int(self._episodes_started),
            "episodes_completed": int(self._episodes_completed),
            "episodes_remaining_to_start": int(max(self._config.num_episodes - self._episodes_started, 0)),
            "active_slots": [
                {
                    "episode_idx": int(slot.episode_idx),
                    "world_idx": int(slot.world_idx),
                    "step_count": int(slot.step_count),
                    "frames_captured": int(len(slot.frames)),
                    "accumulated_reward": float(slot.accumulated_reward),
                    "accumulated_reward_terms": dict(slot.accumulated_reward_terms),
                }
                for slot in active_slots
            ],
            "active_worlds": [int(slot.world_idx) for slot in active_slots],
            "available_render_slots": int(max(self._config.max_parallel_episodes - len(active_slots), 0)),
            "writer_jobs_pending": int(len(self._writer_futures)),
        }

    def on_episode_starts(
        self,
        *,
        world_idx: np.ndarray,
        snapshots_by_world: dict[int, MJWWorldSnapshot],
    ) -> None:
        if self._config is None:
            return

        for raw_world_idx in np.asarray(world_idx, dtype=np.int64).tolist():
            if self._episodes_started >= self._config.num_episodes:
                break
            if len(self._active_slots_by_world) >= self._config.max_parallel_episodes:
                break
            if raw_world_idx in self._active_slots_by_world:
                continue

            snapshot = snapshots_by_world.get(int(raw_world_idx))
            if snapshot is None:
                continue

            episode_idx = self._episodes_started
            slot = _EpisodeSlot(
                world_idx=int(raw_world_idx),
                episode_idx=episode_idx,
            )
            slot.frames.append(
                draw_accumulated_reward(
                    self._render_snapshot(snapshot=snapshot),
                    slot.accumulated_reward,
                    accumulated_reward_terms=slot.accumulated_reward_terms,
                )
            )
            self._active_slots_by_world[int(raw_world_idx)] = slot
            self._episodes_started += 1

        if self._episodes_started >= self._config.num_episodes and not self._active_slots_by_world:
            logger.warning("MJW live recording finished immediately because no episode-start worlds were renderable.")
            self._config = None

    def record_step(
        self,
        *,
        rewards: np.ndarray,
        reward_terms: dict[str, np.ndarray] | None,
        dones: np.ndarray,
        unstable_mask: np.ndarray,
        snapshots_by_world: dict[int, MJWWorldSnapshot],
    ) -> None:
        if self._config is None:
            return

        for world_idx, slot in list(self._active_slots_by_world.items()):
            slot.accumulated_reward += float(rewards[world_idx])
            if reward_terms is not None:
                for label, values in reward_terms.items():
                    slot.accumulated_reward_terms[label] = slot.accumulated_reward_terms.get(label, 0.0) + float(values[world_idx])
            slot.step_count += 1

            done = bool(dones[world_idx])
            unstable = bool(unstable_mask[world_idx])
            should_capture_frame = done or ((slot.step_count % self._config.frame_stride) == 0)

            if should_capture_frame and not unstable:
                snapshot = snapshots_by_world.get(world_idx)
                if snapshot is not None:
                    slot.frames.append(
                        draw_accumulated_reward(
                            self._render_snapshot(snapshot=snapshot),
                            slot.accumulated_reward,
                            accumulated_reward_terms=slot.accumulated_reward_terms,
                        )
                    )

            if done:
                slot.unstable = unstable
                self._finalize_episode(world_idx=world_idx)

    def close(self) -> None:
        if self._active_slots_by_world:
            logger.warning(f"Dropping {len(self._active_slots_by_world)} incomplete MJW recording episode(s) on close.")
        self._active_slots_by_world.clear()
        self._config = None
        self._close_render_context()
        self._cleanup_writer_futures(wait=True)
        self._writer_executor.shutdown(wait=True, cancel_futures=False)

    def _build_camera(self, camera: int | str) -> int | str | mujoco.MjvCamera:
        if camera != -1:
            return camera

        free_camera = mujoco.MjvCamera()
        free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera_config = self._get_default_recording_camera_config()
        free_camera.lookat[:] = np.asarray(camera_config.lookat, dtype=float)
        free_camera.distance = float(camera_config.distance)
        free_camera.azimuth = float(camera_config.azimuth)
        free_camera.elevation = float(camera_config.elevation)
        return free_camera

    def _render_snapshot(self, *, snapshot: MJWWorldSnapshot) -> np.ndarray:
        render_context = self._render_context
        if render_context is None:
            raise RuntimeError("MJW live recording is not active.")

        data = render_context.data
        data.qpos[:] = snapshot.qpos
        data.qvel[:] = snapshot.qvel
        if data.eq_active.size > 0:
            data.eq_active[:] = snapshot.eq_active
        if data.mocap_pos.size > 0:
            data.mocap_pos[:] = snapshot.mocap_pos
        if data.mocap_quat.size > 0:
            data.mocap_quat[:] = snapshot.mocap_quat
        data.time = snapshot.time
        mujoco.mj_forward(self._model, data)
        render_context.renderer.update_scene(data, camera=render_context.camera)
        add_render_geoms = getattr(self._scenario, "add_render_geoms", None)
        if callable(add_render_geoms):
            add_render_geoms(render_context.renderer.scene)
        return np.asarray(render_context.renderer.render()).copy()

    def _finalize_episode(self, *, world_idx: int) -> None:
        if self._config is None:
            return

        slot = self._active_slots_by_world.pop(world_idx)
        self._episodes_completed += 1

        suffix = "_unstable" if slot.unstable else ""
        reward_str = f"{slot.accumulated_reward:.4f}"
        output_path = self._config.video_folder / (
            f"{self._config.video_name_prefix}_ep_{slot.episode_idx}_world_{slot.world_idx}"
            f"-ep_rew={reward_str}{suffix}.mp4"
        )
        frames = slot.frames
        fps = self._effective_fps(self._config)
        future = self._writer_executor.submit(_write_video_file, frames=frames, output_path=output_path, fps=fps)
        self._writer_futures.append(future)
        logger.warning(
            f"Queued MJW recording episode {self._episodes_completed}/{self._config.num_episodes} "
            f"for {output_path.as_posix()} at {fps} fps"
        )

        if self._episodes_completed >= self._config.num_episodes and not self._active_slots_by_world:
            logger.warning(
                f"MJW live recording finished: queued {self._episodes_completed} episode(s) "
                f"to {self._config.video_folder.as_posix()}"
            )
            self._config = None

    def _cleanup_writer_futures(self, *, wait: bool = False) -> None:
        remaining_futures: list[Future[None]] = []
        for future in self._writer_futures:
            if wait:
                future.result()
                continue
            if future.done():
                future.result()
                continue
            remaining_futures.append(future)
        self._writer_futures = remaining_futures

    def _close_render_context(self) -> None:
        render_context = self._render_context
        if render_context is not None:
            render_context.renderer.close()
            self._render_context = None

    @staticmethod
    def _effective_fps(config: MJWRecordingConfig) -> int:
        if config.fps_mode == "fixed":
            return int(config.fps)
        return max(1, int(round(config.fps / config.frame_stride)))

    def _get_default_recording_camera_config(self) -> MJWRecordingCameraConfig:
        scenario_camera_config = self._scenario.get_default_recording_camera_config()
        if scenario_camera_config is not None:
            return scenario_camera_config

        model_center = np.asarray(self._model.stat.center, dtype=float)
        model_extent = float(self._model.stat.extent)
        return MJWRecordingCameraConfig(
            lookat=(float(model_center[0]), float(model_center[1]), float(model_center[2])),
            distance=max(2.0, min(12.0, model_extent * 0.75)),
            azimuth=45.0,
            elevation=-30.0,
        )
