# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:06:11 2024

@author: Sourav
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

# Machine Replacement Class
class River_swim:
    def __init__(self,nS = 6,nA =2):
        self.nS = nS
        self.nA = nA
    def gen_probability(self):
        self.P = np.zeros((self.nA,self.nS,self.nS))
        self.P[0,0,0] = 0.9
        self.P[0,0,1] = 0.1
        self.P[1,self.nS-1,self.nS-1] = 0.9
        self.P[1,self.nS-1,self.nS-2] = 0.1
        for s in range(1,self.nS-1):
            self.P[0,s,s] = 0.6
            self.P[1,s,s] = 0.6
            self.P[0,s,s-1] = 0.3
            self.P[1,s,s-1] = 0.1
            self.P[0,s,s+1] = 0.1
            self.P[1,s,s+1] = 0.3
        self.P[1,0,0] = 0.7
        self.P[1,0,1] = 0.3
        self.P[0,self.nS-1,self.nS-1] = 0.7
        self.P[0,self.nS-1,self.nS-2] = 0.3
        return self.P
    def gen_expected_reward(self):
        self.R = np.zeros((self.nA,self.nS))
        self.R[0,0] = 0.01
        self.R[1,self.nS-1] = 1
        return self.R
    def gen_expected_cost(self):
        self.C = np.zeros((self.nA,self.nS))
        for s in range(self.nS):
            self.C[:,s] = s/10
        return self.C

# Custom Environment
class GymMREnv(gym.Env):
    def __init__(self, mr_obj, init_state, T):
        super(GymMREnv, self).__init__()
        self.P = mr_obj.gen_probability()
        self.R = mr_obj.gen_expected_reward()
        self.C = mr_obj.gen_expected_cost()
        self.init_state = init_state
        self.T = T

        # Environment-specific variables
        self.t = 0
        self.dstate = self.init_state

        # Define observation space and action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_space_size(),), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_space_size())

    def one_hot(self, s):
        one_hot_state = np.zeros(self.observation_space_size())
        one_hot_state[s] = 1
        return one_hot_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.dstate = self.init_state
        self.state = self.one_hot(self.init_state)
        return self.state, {}

    def step(self, action):
        rew = self.R[action, self.dstate]
        cost = self.C[action, self.dstate]
        next_state = np.random.choice(self.P[action, self.dstate, :])
        self.dstate = int(next_state)
        self.t += 1
        done = self.t >= self.T
        trunc = False
        return self.one_hot(int(next_state)), rew, done, trunc, {"cost": cost}

    def observation_space_size(self):
        return len(self.P[0, 0, :])

    def action_space_size(self):
        return self.P.shape[0]

# Initialize Machine Replacement Object
rs_obj = River_swim()
init_state = 3
T = 1000
env = GymMREnv(rs_obj, init_state, T)

# Train a DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

# Evaluate the model
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)