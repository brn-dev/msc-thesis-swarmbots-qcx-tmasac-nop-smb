from swarmbots.utils.mujoco_bootstrap import configure_mujoco_gl_backend

configure_mujoco_gl_backend()

__all__ = ["MJWSwarmBotsVectorEnv"]


def __getattr__(name: str) -> object:
    if name == "MJWSwarmBotsVectorEnv":
        from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv

        return MJWSwarmBotsVectorEnv
    raise AttributeError(name)
