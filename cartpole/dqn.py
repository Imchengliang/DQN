from gettext import translation
import random
from turtle import forward

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.lower = (self.eps_start - self.eps_end) / self.anneal_length
        self.epsilon = self.eps_start

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # print(">>>", x)
        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        sample = random.uniform(0,1)
        
        #self.epsilon -= (self.epsilon - self.eps_end) / self.anneal_length
        
        if self.epsilon > self.eps_end:
            self.epsilon -= self.lower

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (sample >= self.epsilon) or exploit:
            q_values = self.forward(observation)
            with torch.no_grad():
                return torch.max(q_values, 1)[1].view(1,1)

        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        
def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return
    
    """batch - list with tuple: [(obs tuple), (action tuple), (next_obs tuple), (reward tuple)]"""

    batch = memory.sample(dqn.batch_size)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[2])), device=device, dtype=torch.bool)
            
    non_final_next_states = torch.cat([torch.tensor(s).to(device) for s in batch[2] if s is not None])
    obs = torch.cat( batch[0])
    action = torch.cat(batch[1])
    reward = torch.cat(batch[3])
    
    q_values = dqn(obs).gather(1, action)
    
    max_next_q_values = torch.zeros(dqn.batch_size, device=device)
    max_next_q_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
    
    q_value_targets = reward + dqn.gamma * max_next_q_values
    
    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
