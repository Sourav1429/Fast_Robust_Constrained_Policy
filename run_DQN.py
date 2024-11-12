# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:14:33 2024

@author: gangu
"""
import numpy as np
import gym
from DQN import DQNAgent


env = gym.make('CartPole-v1');
state_size,action_size = env.observation_space.shape[0],env.action_space.n
seed = 0
agent = DQNAgent(state_size, action_size, seed)
n_episode,max_step,eps,eps_end,eps_decay = 10,1000,1.0,0.01,0.995
rewards,scores = [],[]
for i_episode in range(n_episode):
    print('Episode=',i_episode)
    state = env.reset()[0]
    score = 0
    eps = max(eps_end,eps_decay*eps)
    for t in range(max_step):
        action = agent.act(state,eps)
        next_state,reward,done,trunc,_ = env.step(action)
        agent.step(state,action,reward,next_state,done)
        state = next_state
        score+=reward
        if done or trunc:
            break
        print(f"\tScore:{score},epsilon:{eps}")
        rewards.append(score)
        scores.append(np.mean(rewards[-100:]))
env.close()