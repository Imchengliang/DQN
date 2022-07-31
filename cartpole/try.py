import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


env = gym.make("CartPole-v0")
def Random_games():
    for episode in range(10):
        env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            print(">>>>",action)
            next_state, reward, done, info = env.step(action)
            print(t, next_state, reward, done, info, action)
            if done:
                break
Random_games()
