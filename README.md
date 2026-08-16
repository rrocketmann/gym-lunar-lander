# Gymnasium Lunar Lander DQN

This repository contains two different approaches to solving the Gymnasium Lunar Lander environment: a simple rule-based algorithm and a Machine Learning (Reinforcement Learning) model.

## Demonstration
![](https://github.com/rrocketmann/gym-lunar-lander/blob/main/landing.gif)
## 1. Machine Learning Model (DQN)

The ML implementation uses a Q-Network algorithm. 

### Usage:
- **Train the Model (Headless):** 
  ```bash
  python main.py
  ```
  This trains the model over 1000 fast episodes and saves the progress to `lunar_lander_model.pth`.

---

## 2. Rule-Based Algorithm (If-Statement Logic)

This was the initial approach containing hand-tuned conditions (`if/elif/else`) that manually monitor position, velocity, and tilt to guide the module.

### Usage:
- **Watch the Rule-Based Control:**
  ```bash
  python rule_based.py
  ```
