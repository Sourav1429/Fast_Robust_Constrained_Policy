# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:08:14 2024

@author: gangu
"""

# import gym
# from gym import spaces
# import numpy as np
from Machine_Rep import MachineReplacementEnv,RiverSwimEnv
import torch as th
from torch import nn
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor

env_set = ["MR","RS"]
env_choice = 0
env =  None
if(env_choice ==0):
    env = MachineReplacementEnv()
else:
    env = RiverSwimEnv()



class CustomQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_space.n, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)  # Output raw Q-values
        )

    def forward(self, x):
        q_values = self.fc(x)
        action_probs = th.softmax(q_values, dim=1)  # Convert Q-values to probabilities
        return action_probs

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)
        self.mlp_extractor = CustomQNetwork(self.observation_space, self.action_space)

    def forward(self, obs, deterministic=False):
        obs = obs_as_tensor(obs, self.device)
        action_probs = self.mlp_extractor(obs)
        if deterministic:
            actions = th.argmax(action_probs, dim=1)
        else:
            actions = th.multinomial(action_probs, num_samples=1)
        return actions, None

model = DQN(
    policy=CustomPolicy,
    env=env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=100,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1,
)
model.learn(total_timesteps=10)




