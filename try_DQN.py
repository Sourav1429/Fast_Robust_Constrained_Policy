# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:28:48 2024

@author: gangu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import gymnasium as gym
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
'''
# Hyperparameters
state_size = 4                # For CartPole-v1 environment
action_size = 2               # Number of actions
batch_size = 64               # Batch size for experience replay
gamma = 0.99                  # Discount factor
learning_rate = 0.001         # Learning rate for optimizer
epsilon_start = 1.0           # Initial exploration rate
epsilon_end = 0.01            # Minimum exploration rate
epsilon_decay = 0.995         # Decay rate for epsilon

# Initialize environment and DQN
env = gym.make('CartPole-v1')
qnetwork = DQN(state_size, action_size)
target_network = DQN(state_size, action_size)
target_network.load_state_dict(qnetwork.state_dict())
target_network.eval()

optimizer = optim.Adam(qnetwork.parameters(), lr=learning_rate)
memory = deque(maxlen=2000)
epsilon = epsilon_start

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = qnetwork(state)
            return q_values.argmax().item()

def experience_replay(batch_size):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    # Compute Q(s, a) with the local Q-network
    q_values = qnetwork(states).gather(1, actions).squeeze()

    # Compute max Q(s', a) with the target Q-network
    next_q_values = target_network(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Loss
    loss = nn.functional.mse_loss(q_values, target_q_values.detach())
    
    # Optimize the Q-network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

num_episodes = 1000
target_update_frequency = 10

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for t in range(200):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Store experience in replay memory
        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        
        # Perform experience replay and train the network
        experience_replay(batch_size)
        
        if done:
            break
    
    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    # Update the target network
    if episode % target_update_frequency == 0:
        target_network.load_state_dict(qnetwork.state_dict())
    
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()
torch.save(target_network,"target_dqn_model")
torch.save(qnetwork,"qnetwork_dqn_model")
'''
