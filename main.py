from enum import Enum
import gymnasium as gym
from random import choice
import pygame

pygame.init()

class Action(Enum):
    NOTHING = 0
    LEFT_ENGINE = 1
    MAIN_ENGINE = 2
    RIGHT_ENGINE = 3

total_reward = 0.0
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()
# actions = [0, 1, 2, 3]

episode_over = False
while not episode_over:
    linear = (observation[0], observation[1])
    linear_vel = (observation[2], observation[3])
    angular = observation[4]
    angular_vel = observation[5]

    # action_id = choice(actions)
    if linear_vel[1] < -0.4 and abs(angular) < 1:
        action_id = Action.MAIN_ENGINE.value
        print("MAIN ENGINE")
    elif angular > 0.2:
        action_id = Action.RIGHT_ENGINE.value
        print("RIGHT ENGINE")
    elif angular < -0.2:
        action_id = Action.LEFT_ENGINE.value
        print("LEFT ENGINE")
    elif linear[0] < -0.1:
        action_id = Action.RIGHT_ENGINE.value
        print("RIGHT ENGINE")
    elif linear[0] > 0.1:
        action_id = Action.LEFT_ENGINE.value
        print("LEFT ENGINE")
    elif linear_vel[0] < -0.1:
        action_id = Action.RIGHT_ENGINE.value
        print("RIGHT ENGINE")
    elif linear_vel[0] > 0.1:
        action_id = Action.LEFT_ENGINE.value
        print("LEFT ENGINE")
    else:
        action_id = Action.NOTHING.value
        print("NOTHING")
    print(total_reward)

    observation, reward, terminated, truncated, info = env.step(action_id)
    total_reward += float(reward)
    episode_over = terminated or truncated

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        break

env.close()
print("Total Reward:", total_reward)
