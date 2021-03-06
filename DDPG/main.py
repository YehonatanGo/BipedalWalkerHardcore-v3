import gym
import matplotlib.pyplot as plt

from DDPG.ddpg import *


if __name__ == "__main__":
  env = gym.make("BipedalWalkerHardcore-v3")
  # env = wrap_env(env)
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.shape[0]
  agent = DDPGAgent(state_size, action_size, 0)

  rewards = agent.solve(env, 1000)

  plt.title("DDPG")
  plt.xlabel("Episodes")
  plt.ylabel("Rewards")
  plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
  plt.show()
  plt.savefig('DDPG.png')