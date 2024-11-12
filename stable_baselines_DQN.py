# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:31:57 2024

@author: gangu
"""

import gymnasium as gym

from stable_baselines3 import DQN
import numpy as np
from PIL import Image

def create_gif(frames, output_path="output.gif", duration=5000, loop=0):
    """
    Create a GIF from a list of image frames.

    Parameters:
        frames (list): List of frames (each frame should be a PIL Image or NumPy array).
        output_path (str): File path to save the GIF.
        duration (int): Duration of each frame in milliseconds.
        loop (int): Number of times the GIF should loop (0 for infinite).

    Returns:
        None
    """
    # Ensure all frames are PIL Images
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]

    # Save frames as a GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved at {output_path}")

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = DQN("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10000, log_interval=4)
#model.save("dqn_cartpole")

#del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs, info = env.reset()
frames=[]
n_steps = 200
i=-1
rews=0
while True:
    i+=1
    action, _states = model.predict(obs, deterministic=True)
    frame = env.render()
    frames.append(frame)
    obs, reward, terminated, truncated, info = env.step(action)
    rews+=reward
    if terminated or truncated or i<n_steps:
        break
last_fr = np.zeros((len(frames[0]),len(frames[0][0])))
frames.append(last_fr)
create_gif(frames)
print("Total rewards",rews)