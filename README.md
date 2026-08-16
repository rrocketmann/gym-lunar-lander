# Gymnasium Lunar Lander

This repository contains two approaches for `LunarLander-v3` in Gymnasium:

- A tabular RL agent in [lander.py](lander.py)
- A simple rule-based controller in [rule_lander.py](rule_lander.py)

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Tabular RL Agent

The RL implementation in [lander.py](lander.py) trains in a headless environment for speed, then runs evaluation and visible rollout episodes.

Run with defaults:

```bash
python lander.py
```

Default behavior:

- Trains for `10000` episodes
- Prints test metrics over `1000` episodes
- Opens a visual window for `3` rollout episodes (`render_mode="human"`)

## 2. Rule-Based Controller

Run the hand-written policy with human rendering:

```bash
python rule_lander.py
```
