# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:24:20 2024

@author: gangu
"""

import numpy as np
from Machine_Rep import Machine_Replacement,gym_MR_env
from try_DQN import DQN


if __name__ == "__main__":
    mr_obj = Machine_Replacement()
    init_state = 0
    T = 1000
    env = gym_MR_env(mr_obj,init_state,T)
    state_size,action_size = env.observation_space_size,env.action_space_size()
    seed = 0
    batch_size = 64               # Batch size for experience replay
    gamma = 0.99                  # Discount factor
    learning_rate = 0.001         # Learning rate for optimizer
    epsilon_start = 1.0           # Initial exploration rate
    epsilon_end = 0.01            # Minimum exploration rate
    epsilon_decay = 0.995 
    
    qnetwork = DQN(state_size, action_size)