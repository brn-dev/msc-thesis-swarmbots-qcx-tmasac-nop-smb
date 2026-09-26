# SwarmBots — Thesis Repository

This repository accompanies the [thesis](thesis.pdf) and releases the current
state of the SwarmBots simulation environments and multi-agent reinforcement
learning algorithms. It contains the environment implementations, learning
code, experiment configurations, and analysis tools used in the thesis, along
with research extensions and follow-up experiments.

SwarmBots is a continuous-control multi-agent reinforcement learning benchmark
for modular robots. Identical articulated units can move independently and
create or release load-bearing connections during an episode. A policy therefore
controls both the agents' motion and the changing morphology of the collective.

The thesis investigates scalable learning for these dynamic, partially
observable swarms. The code includes tasks for locomotion, obstacle traversal,
spatial exploration, and payload transport, together with training and
evaluation code for several multi-agent learning architectures.

## Components

- **Simulation environments** (`swarmbots/mj_env`, `swarmbots/mjw_env`): an
  interactive CPU MuJoCo environment for development and visualization, plus a
  batched MuJoCo Warp environment for large-scale GPU training.
- **Swarm embodiment and connectivity**: configurable articulated modules,
  swarm generation, and controllable mechanical connectors that act as dynamic
  edges in the swarm's physical graph.
- **Scenario suite** (`swarmbots/*_env/scenarios`,
  `swarmbots/scenario_presets`): navigation, wall and partially observable wall
  traversal, opening discovery, climbing and bridging, and single- or
  multi-payload transport tasks.
- **Learning stack** (`swarmbots/learn`): PPO and MAPPO baselines,
  Multi-Agent Transformer variants, Transformer-based Multi-Agent Soft
  Actor-Critic (TMASAC), recurrent policies, replay and rollout infrastructure,
  environment wrappers, checkpointing, metrics, and evaluation utilities.
- **Research extensions**: typed multi-step next-observation prediction (NOP)
  and bounded multimodal action distributions, including Signed-Magnitude Beta
  (SMB).
- **Experiments and analysis** (`experiments`, `scripts`): thesis experiment
  configurations, ablations, morphology-transfer and connector studies,
  evaluation/recording entry points, plotting utilities, and throughput
  benchmarks.

## Environment

The project is packaged with `uv`; dependencies and optional feature groups are
defined in `pyproject.toml`. The current package targets Python 3.13. Individual
research runs are launched from the entry-point scripts under `experiments/` or
`scripts/`; these should be treated as experiment specifications rather than a
stable command-line interface.
