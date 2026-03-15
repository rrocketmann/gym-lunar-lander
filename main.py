import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, obs_size=8, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_size=8, n_actions=4, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.q_net = QNetwork(obs_size, n_actions)
        self.target_net = QNetwork(obs_size, n_actions)  # stable target
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=50_000)  # replay buffer

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def get_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(obs_t).argmax().item()

    def store(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs_t      = torch.FloatTensor(np.array(obs))
        next_obs_t = torch.FloatTensor(np.array(next_obs))
        actions_t  = torch.LongTensor(np.array(actions))
        rewards_t  = torch.FloatTensor(np.array(rewards))
        dones_t    = torch.FloatTensor(np.array(dones))

        current_q  = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(next_obs_t).max(1).values
            target_q   = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


import sys
import gymnasium as gym

def train():
    print("Starting fast training (no visuals)...")
    env = gym.make("LunarLander-v3")  # Removed render_mode="human" to speed up
    agent = DQNAgent()

    N_EPISODES = 1000  # Increased for more reliable convergence
    TARGET_SYNC_EVERY = 10  # copy q_net → target_net every N episodes

    for episode in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            agent.train_step()

            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()

        if episode % TARGET_SYNC_EVERY == 0:
            agent.sync_target()

        # Only print every 10 episodes to reduce terminal clutter
        if episode % 10 == 0:
            print(f"Episode {episode:>4}  reward: {total_reward:>8.1f}  epsilon: {agent.epsilon:.3f}")

    env.close()
    
    # Save the model
    torch.save(agent.q_net.state_dict(), "lunar_lander_model.pth")
    print("\nTraining complete! Exported model to lunar_lander_model.pth")


def play():
    print("Loading saved model and visualizing...")
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent()
    
    try:
        agent.q_net.load_state_dict(torch.load("lunar_lander_model.pth", weights_only=True))
    except FileNotFoundError:
        print("Error: 'lunar_lander_model.pth' not found. Please train the model first by running `python main.py`")
        return

    agent.q_net.eval()
    agent.epsilon = 0.0  # Turn off random exploration during testing

    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
            
        print(f"Test Episode {episode+1} reward: {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play()
    else:
        train()