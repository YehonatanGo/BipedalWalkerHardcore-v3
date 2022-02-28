import gym
import matplotlib.pyplot as plt

from D3QN.d3qn import *


if __name__ == "__main__":
  env = gym.make("BipedalWalkerHardcore-v3")
  # env = wrap_env(env)
  state_size = env.observation_space.shape[0]
  action_size = len(discrete_actions)
  agent = D3QNAgent(state_size, action_size, 0)

  rewards = agent.solve(env, 3000)

  plt.title("D3QN, lr = 0.00008, episodial eps decay")
  plt.xlabel("Episodes")
  plt.ylabel("Rewards")
  plt.plot([i + 1 for i in range(0, len(rewards), 2)], rewards[::2])
  plt.savefig('D3QN.png')
  plt.show()
