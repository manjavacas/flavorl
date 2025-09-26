# MEARL 🤖🍽️

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-compatible-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MEARL is a reinforcement learning (RL) environment designed to simulate interactions in meal recommendation systems, specifically based on the [MealRec+](https://github.com/WUT-IDEA/MealRecPlus) dataset. This environment enables training and evaluating RL agents in the domain of food recommendations.

## 📦 Installation

```bash
git clone https://github.com/manjavacas/mearl.git
cd mearl
pip install -e .
```

## 🎯 Quick start

```python
import gymnasium as gym
import mearl

# Create the environment
env = gym.make("mearl/MealRecPlus-v0", obs_dim=20, act_dim=10)

# Run an episode
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## 📚 Citation

If you use MEARL in your research, please cite:

```bibtex
...
```

## 👥 Authors

- **Antonio Manjavacas** - [@manjavacas](https://github.com/manjavacas)
- **Andrea Morales** - [@andreamorgar](https://github.com/andreamorgar)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) for details.