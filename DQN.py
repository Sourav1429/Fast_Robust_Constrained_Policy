# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:44:49 2024

@author: gangu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self,state_size,action_size):
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_size)
        
    def forward(self,x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    
class Replay_Buffer:
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
    def push(self,state,action,reward,next_state,done):
        if len(self.buffer)<self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state,action,reward,next_state,done)
        self.index = (self.index+1)%self.capacity
    def sample(self,batch_size):
        batch = np.random.choice(len(self.buffer),batch_size,replace = False)
        states,actions,rewards,next_states,dones = [],[],[],[],[]
        for i in batch:
            state,action,reward,next_state,done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return(
            torch.tensor(np.array(states)).float(),
            torch.tensor(np.array(actions)).float(),
            torch.tensor(np.array(rewards)).unsqueeze(1).float(),
            torch.tensor(np.array(next_states)).float(),
            torch.tensor(np.array(dones)).unsqueeze(1).int())
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self,state_size,action_size,seed,learning_rate=1e-3,capacity=100000,discount_fact=0.9,update_every=4,batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.learning_rate = learning_rate
        self.discount_factor = discount_fact
        self.update_every = update_every
        self.batch_size = batch_size
        self.steps = 0
        self.Qnetwork_local = QNetwork(state_size,action_size)
        self.optimizer = optim.Adam(self.Qnetwork_local.parameters(),lr = learning_rate)
        self.replay_buffer = Replay_Buffer(capacity)
    
    def step(self,state,action,reward,next_state,done):
        self.replay_buffer.push(state,action,reward,next_state,done)
        self.steps+=1
        if self.steps%self.update_every == 0:
            if self.replay_buffer.__len__() > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.learn(experiences)
                
    def act(self,state,eps=0.1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if random.random()>eps:
           state = torch.tensor(state).float().unsqueeze(0).to(device)
           self.Qnetwork_local.eval()
           with torch.no_grad():
               action_values = self.Qnetwork_local(state)
           return np.argmax(action_values.cpu().data.numpy())
        else:
           return random.choice(np.arange(self.action_size))
    
    def learn(self,experiences):
        states,actions,rewards,next_states,dones = experiences
        Q_targets_next = self.Qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.discount_factor*Q_targets_next
        print(actions.view(-1,1).int())
        Q_expected = self.Qnetwork_local(states).gather(1,actions.view(-1,1))
        loss = F.mse_loss(Q_expected,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
            
                
    
    
        

