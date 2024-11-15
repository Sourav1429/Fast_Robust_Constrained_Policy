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
    n_episode,max_step,eps,eps_end,eps_decay = 1000,1000,1.0,0.01,0.995
    rewards,scores = [],[]
    score = 0
    for i_episode in range(n_episode):
        print('Episode=',i_episode)
        state = env.reset()
        score = 0
        eps = max(eps_end,eps_decay*eps)
        for t in range(max_step):
            action = agent.act(state,eps)
            next_state,reward,done,trunc,_ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score+=reward
            if done or trunc :
                break
            #print(f"\tScore:{score},epsilon:{eps}")
            rewards.append(score)
            scores.append(np.mean(rewards[-100:]))
        print(f"Episode ran for {t} steps and final score is {score}")
    env.close()
