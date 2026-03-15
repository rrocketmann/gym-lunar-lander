# Gymnasium Lunar Lander DQN

This repository contains two different approaches to solving the Gymnasium Lunar Lander environment: a simple rule-based algorithm and a Machine Learning (Reinforcement Learning) model.

## Demonstration

<video src="./example.webm" width="600" controls autoplay loop></video>

## 1. Machine Learning Model (DQN)

The ML implementation uses a Deep Q-Network (`DQN`) algorithm written with PyTorch. 

### Usage:
- **Train the Model (Headless):** 
  ```bash
  python main.py
  ```
  This trains the model over 1000 fast episodes and saves the progress to `lunar_lander_model.pth`.

- **Watch the Trained Model (Visual):**
  ```bash
  python main.py play
  ```
  This will open a render window and let the built-in trained agent attempt to land the craft based on its network weights.

---

## 2. Rule-Based Algorithm (If-Statement Logic)

This was the initial approach containing hand-tuned conditions (`if/elif/else`) that manually monitor position, velocity, and tilt to guide the module.

### Usage:
- **Watch the Rule-Based Control:**
  ```bash
  python rule_based.py
  ```
