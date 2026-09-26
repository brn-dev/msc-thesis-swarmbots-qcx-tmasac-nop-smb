
import collections.abc
import typing
from collections import OrderedDict
from collections.abc import KeysView, Sequence
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium.spaces.space import Space

# partially from gymnasium.spaces.Dict
class HybridActionSpace(Space[dict[str, Space[Any]]], typing.Mapping[str, Space[Any]]):

    def __init__(
        self,
        spaces: dict[str, Space] | Sequence[tuple[str, Space]],
        seed: int | np.random.Generator | None = None,
    ):
        if isinstance(spaces, OrderedDict):
            spaces = spaces.copy()
        elif isinstance(spaces, collections.abc.Mapping):
            spaces = OrderedDict(sorted(spaces.items()))
        elif isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)
        else:
            raise TypeError(
                f"Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: {type(spaces)}"
            )

        if len(spaces) == 0:
            raise ValueError("HybridActionSpace requires at least one sub-space.")

        self.n_spaces = len(spaces)
        self.space_map: OrderedDict[str, Space[Any]] = spaces
        self.key_order = list(self.space_map.keys())
        self.sub_spaces = list(self.space_map.values())
        self.agent_action_dims = [get_agent_action_dim(s) for s in self.space_map.values()]
        self.total_agent_action_dim = sum(self.agent_action_dims)
        self.n_agents = self.sub_spaces[0].shape[0]
        for sub_space in self.sub_spaces[1:]:
            if sub_space.shape[0] != self.n_agents:
                raise ValueError(
                    f"All sub-spaces must share n_agents in shape[0]: expected {self.n_agents}, got {sub_space.shape[0]}"
                )
        
        super().__init__(None, None, seed)

    @property
    def spaces(self) -> OrderedDict[str, Space[Any]]:
        return self.space_map

    def concat_actions(self, actions_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            tuple(actions_dict[k] for k in self.key_order),
            axis=-1
        )

    def split_actions(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        split = np.split(actions, np.cumsum(self.agent_action_dims)[:-1], axis=-1)
        return {k: v for k, v in zip(self.key_order, split, strict=True)}

    def sample(
        self,
        mask: dict[str, Any] | None = None,
        probability: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generates a single random sample from this space.

        The sample is an ordered dictionary of independent samples from the constituent spaces.

        Returns:
            A dictionary with the same key and sampled values from :attr:`self.spaces`
        """
        if mask is not None or probability is not None:
            raise NotImplementedError('todo')
        return {k: space.sample() for k, space in self.space_map.items()}

    def contains(self, x: Any) -> bool:
        raise NotImplementedError()

    def __getitem__(self, key: str) -> Space[Any]:
        return self.space_map[key]

    def keys(self) -> KeysView:
        return KeysView(self.space_map)

    def __setitem__(self, key: str, value: Space[Any]):
        raise NotImplementedError()

    def __iter__(self):
        yield from self.space_map

    def __len__(self) -> int:
        return len(self.space_map)

    def __repr__(self) -> str:
        return (
            "HybridSpace(" + ", ".join([f"{k!r}: {s}" for k, s in self.space_map.items()]) + ")"
        )

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError()

    def to_jsonable(self, sample_n: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
        raise NotImplementedError()

    def from_jsonable(self, sample_n: dict[str, list[Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError()

def get_agent_action_dim(space: Space) -> int:
    """
    Return the per-agent action dimension for an action space.

    Assumption: Each individual action component space is shaped like (n_agents, n_actions_per_agent)
    """
    if isinstance(space, (gym.spaces.Box, gym.spaces.MultiBinary)):
        shape = getattr(space, "shape", None)
        if shape is None:
            raise ValueError(f"Expected a shaped space (n_agents, n_actions_per_agent), got {space}")
        if len(shape) != 2:
            raise ValueError(f"Expected shape (n_agents, n_actions_per_agent), got shape={shape} for {space}")
        return int(shape[1])
    raise NotImplementedError(f"{space} space is not supported (expected Box/MultiBinary)")


class VectorHybridActionSpace(HybridActionSpace):
    def __init__(
        self,
        spaces: dict[str, Space] | Sequence[tuple[str, Space]],
        seed: int | np.random.Generator | None = None,
    ):
        if isinstance(spaces, OrderedDict):
            spaces = spaces.copy()
        elif isinstance(spaces, collections.abc.Mapping):
            spaces = OrderedDict(sorted(spaces.items()))
        elif isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)
        else:
            raise TypeError(
                f"Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: {type(spaces)}"
            )

        if len(spaces) == 0:
            raise ValueError("VectorHybridActionSpace requires at least one sub-space.")

        self.n_spaces = len(spaces)
        self.space_map: OrderedDict[str, Space[Any]] = spaces
        self.key_order = list(self.space_map.keys())
        self.sub_spaces = list(self.space_map.values())

        self.agent_action_dims = [get_vector_agent_action_dim(s) for s in self.space_map.values()]
        self.total_agent_action_dim = sum(self.agent_action_dims)

        # (n_envs, n_agents, dim)
        self.n_envs = int(self.sub_spaces[0].shape[0])
        self.n_agents = int(self.sub_spaces[0].shape[1])
        for sub_space in self.sub_spaces[1:]:
            if int(sub_space.shape[0]) != self.n_envs:
                raise ValueError(
                    f"All sub-spaces must share n_envs in shape[0]: expected {self.n_envs}, got {sub_space.shape[0]}"
                )
            if int(sub_space.shape[1]) != self.n_agents:
                raise ValueError(
                    f"All sub-spaces must share n_agents in shape[1]: expected {self.n_agents}, got {sub_space.shape[1]}"
                )

        Space.__init__(self, None, None, seed)


def get_vector_agent_action_dim(space: Space) -> int:
    """
    Return the per-agent action dimension for a *vector* action space.

    Assumption: Each individual action component space is shaped like (n_envs, n_agents, n_actions_per_agent)
    """
    if isinstance(space, (gym.spaces.Box, gym.spaces.MultiBinary)):
        shape = getattr(space, "shape", None)
        if shape is None:
            raise ValueError(f"Expected a shaped space (n_envs, n_agents, n_actions_per_agent), got {space}")
        if len(shape) != 3:
            raise ValueError(f"Expected shape (n_envs, n_agents, n_actions_per_agent), got shape={shape} for {space}")
        return int(shape[2])
    raise NotImplementedError(f"{space} space is not supported (expected Box/MultiBinary)")

