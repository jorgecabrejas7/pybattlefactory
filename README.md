# PyBattle: Pokemon Emerald Battle Simulator

A high-performance C++ Pokemon battle simulator with Python bindings and Gymnasium integration, optimized for Reinforcement Learning.

## Features

- **Fast & Parallelizable**: Core logic implemented in C++ for maximum throughput.
- **Accurate Mechanics**: Faithful replication of Gen 3 (Emerald) battle logic, AI scripts, and RNG.
- **Gymnasium Ready**: Includes standard `gym.Env` wrappers for combat and Hierarchical RL.
- **Battle Factory Focused**: Specialized support for Battle Factory challenge cycles (Rental, Battle, Swap).
- **Hierarchical RL Support**: Built-in phase management and action masking for multi-agent systems.

## Installation

### Prerequisites

- C++17 compatible compiler (GCC, Clang, or MSVC)
- CMake 3.16+
- Python 3.8+

### Install from Source

```bash
git clone https://github.com/your-username/pokeemerald-simulator.git
cd pokeemerald-simulator/simulator
pip install .
```

For development:
```bash
pip install -e .
```

## Usage

### Standard Combat Environment

```python
import pybattle
import gymnasium as gym

env = pybattle.PokemonEnv(challenge_num=0, is_open_level=True, seed=42)
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Hierarchical Battle Factory Environment (HRL)

```python
from pybattle import FactoryHRL_Env, FactoryPhase

env = FactoryHRL_Env(challenge_num=0, is_open_level=True)
obs, info = env.reset()

# The environment handles:
# 1. RENTAL Phase (picking lead team)
# 2. BATTLE Phase (turn-by-turn combat)
# 3. SWAP Phase (exchanging team members after a victory)

phase = obs[0] # FactoryPhase.RENTAL
mask = obs[80:100] # Legal action mask
```

## Project Structure

- `src/`: C++ Core simulator source code.
- `include/`: C++ Header files.
- `pybattle/`: Python package source and Gymnasium wrappers.
- `tests/`: C++ unit tests for battle logic and AI.

## License

MIT
