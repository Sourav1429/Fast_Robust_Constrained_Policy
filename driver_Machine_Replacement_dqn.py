# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:24:20 2024

@author: gangu
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Machine_Rep import Machine_Replacement,gym_MR_env
from try_DQN import DQN
import random
from collections import deque


if __name__ == "__main__":
    mr_obj = Machine_Replacement()
    init_state = 0
    T = 1000
    env = gym_MR_env(mr_obj,init_state,T)
    state_size,action_size = env.observation_space_size(),env.action_space_size()
    seed = 0
    batch_size = 64               # Batch size for experience replay
    gamma = 0.99                  # Discount factor
    learning_rate = 0.001         # Learning rate for optimizer
    epsilon_start = 1.0           # Initial exploration rate
    epsilon_end = 0.01            # Minimum exploration rate
    epsilon_decay = 0.995 
    
    qnetwork = DQN(state_size, action_size)
    target_network = DQN(state_size, action_size)
    target_network.load_state_dict(qnetwork.state_dict())
    target_network.eval()
    optimizer = optim.Adam(qnetwork.parameters(), lr=learning_rate)
    memory = deque(maxlen=2000)
    epsilon = epsilon_start
    def select_action(state, epsilon):
      if random.random() < epsilon:
          #print("No problem 1")
          return np.random.choice(action_size)  # Random action
      else:
          with torch.no_grad():
              #print("Printing state")
              #print(state)
              state = torch.FloatTensor(state).unsqueeze(0)
              q_values = qnetwork(state)
              #print("no problem 2")
      return q_values.argmax().item()
    
    def experience_replay(batch_size):
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        #print("reached here")
        #print(states)
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
    
    for episode in range(num_episodes):#changed num_episodes to 2
        state = env.reset()
        total_reward = 0
        #print("One step start")
        for t in range(200):#changed 200 to 2
            action = select_action(state, epsilon)
            #print("Here is the problem1")
            cost, next_state, terminated, truncated,utility, _ = env.step(action)
            #print("Here is the problem2")
            reward = -cost
            done = terminated or truncated
            total_reward += reward
            
            # Store experience in replay memory
            #print("Printing state")
            #print(state)
            #break
            #print("Printing action")
            #print(action)
            #break
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
    
    #env.close()
    torch.save(target_network,"target_dqn_model_Machine_Replacement")
    torch.save(qnetwork,"qnetwork_dqn_model_Machine_Replacement")
