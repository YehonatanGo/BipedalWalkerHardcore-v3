import gym
import torch
import argparse
import numpy as np

from utils import *
from sac import SACAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    env = gym.make("BipedalWalkerHardcore-v3")
    torch.manual_seed(0)
    np.random.seed(0)
    env.seed(0)
    # env = wrap_env(env)
    max_steps = env._max_episode_steps
    print(max_steps)
    state_size = env.observation_space.shape[0]
    action_space = env.action_space
    hidden_size = 256
 

    agent = SACAgent(state_size, action_space, hidden_size)
    agent.solve(env)
    agent.save_model('models','final')
