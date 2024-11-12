# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:14:33 2024

@author: gangu
"""
import numpy as np
import gymnasium as gym
from DQN import DQNAgent    
from PIL import Image

def create_gif(frames, output_path="output.gif", duration=100, loop=0):
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

env = gym.make('CartPole-v1',render_mode = "rgb_array");
state_size,action_size = env.observation_space.shape[0],env.action_space.n
seed = 0
agent = DQNAgent(state_size, action_size, seed)
n_episode,max_step,eps,eps_end,eps_decay = 1000,1000,1.0,0.01,0.995
rewards,scores = [],[]
frame_capture = 10
frames=[]
score = 0
for i_episode in range(n_episode):
    print('Episode=',i_episode)
    state = env.reset()[0]
    score = 0
    eps = max(eps_end,eps_decay*eps)
    for t in range(max_step):
        action = agent.act(state,eps)
        if i_episode == n_episode-1:
          frame = env.render()
          #print(frame)
          frames.append(frame)
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
#create_gif(frames)