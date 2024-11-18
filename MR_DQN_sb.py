# Import necessary libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

# Machine Replacement Class
class Machine_Replacement:
    def __init__(self, rep_cost=0.7, safety_cost=0.5, nS=4, nA=2):
        self.nS = nS
        self.nA = nA
        self.cost = np.linspace(0.1, 0.99, nS)
        self.rep_cost = rep_cost
        self.safety_cost = safety_cost

    def gen_probability(self):
        self.P = np.zeros((self.nA, self.nS, self.nS))
        for i in range(self.nS):
            for j in range(self.nS):
                if i <= j:
                    self.P[0, i, j] = (i + 1) * (j + 1)
                else:
                    continue
            self.P[0, i, :] = self.P[0, i, :] / np.sum(self.P[0, i, :])
            self.P[1, i, 0] = 1
        return self.P

    def gen_reward(self):
        self.R = np.zeros((self.nA, self.nS, self.nS))
        for i in range(self.nS):
            self.R[0, i, :] = self.cost[i]
            self.R[1, i, 0] = self.rep_cost + self.cost[0]
        return self.R

    def gen_expected_reward(self):
        self.R = np.zeros((self.nA, self.nS))
        for i in range(self.nS):
            self.R[0, i] = self.cost[i]
            self.R[1, i] = self.rep_cost + self.cost[0]
        return self.R

    def gen_expected_cost(self):
        self.C = np.zeros((self.nA, self.nS))
        for i in range(self.nS):
            self.C[0, i] = self.cost[i]
            self.C[1, i] = self.safety_cost + self.cost[0]
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
mr_obj = Machine_Replacement(rep_cost=0.7, safety_cost=0.5, nS=4, nA=2)
init_state = 0
T = 10
env = GymMREnv(mr_obj, init_state, T)

# Train a DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)

# Evaluate the model
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward

print("Total Reward:", total_reward)
