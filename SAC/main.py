import argparse
import time
import gym
import numpy as np
import torch
from sac import soft_actor_critic_agent
from memory import ReplayMemory
from collections import deque
import matplotlib.pyplot as plt
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-Soft-Actor-Critic
'''



def solve(args,max_steps,env,agent,memory):

    total_numsteps = 0
    updates = 0
    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = [] 
    episodes_array = []
    
    for i_episode in range(args.iteration): 
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        for step in range(max_steps):    
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Update parameters of all the networks
                agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            if reward == -100:
                reward = -1
            # reward = reward * 10
            env.render()
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == 2000 else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            state = next_state
            
            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)        
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        episodes_array.append(i_episode)

        print(f"Episode: {i_episode}, Score: {episode_reward}, Avg.Score: {avg_score}", end="\r")
        if (avg_score >= 300):
            print('Solved environment with Avg Score:  ', avg_score)
            plt.title("SAC")
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.plot([i + 1 for i in range(0, len(scores_array), 2)], scores_array[::2])
            plt.savefig("SAC.jpg")
            plt.show()
            break
            
    return scores_array, avg_scores_array 



if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--capacity',       default = 1000000,            type = int,   help = ' Size of replay buffer') 
    parser.add_argument('--iteration',      default = 100000,             type = int,   help = 'Num of episodes')
    parser.add_argument('--batch_size',     default = 256,                type = int,   help = 'Mini batch size') 
    parser.add_argument('--seed',           default = 0,                  type = int,   help = 'Random seed number')
    parser.add_argument('--learning_rate',  default = 0.00008,            type = float, help = 'Learning rate of optimizer')
    parser.add_argument('--gamma',          default = 0.99,               type = float, help = 'Discount factor')
    parser.add_argument('--hidden_size',    default = 256,                type = int,   help = 'Hidden size of net')    
    parser.add_argument('--alpha',          default = 0.2,                type = float, help = 'Temperature parameter α determines the relative importance of the entropy term against the reward')
    parser.add_argument('--tau',            default = 0.005,              type = float, help = 'Target smoothing coefficient(τ)')
    parser.add_argument('--start_steps',    default = 10000,              type = int,   help = 'Steps sampling random actions')
    parser.add_argument('--rd_intl',        default = 20,                 type = int,   help = 'Interval of render')
    parser.add_argument('--render',         default = True,               type = bool,  help = 'Shoe UI animation') 
    parser.add_argument('--train',          default = True,               type = bool,  help = 'Train the model') 
    parser.add_argument('--eval',           default = False,              type = bool,  help = 'Evaluate the model') 
    parser.add_argument('--load',           default = False,              type = bool,  help = 'Load trained model') 
    parser.add_argument('--directory',      default = 'models',           type = str,   help = 'Directory for saving actor-critic model') 

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make("BipedalWalkerHardcore-v3")
    env.seed(args.seed)
    max_steps = 2000
    env = wrap_env(env)

    agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, 
                                    device=device, hidden_size=args.hidden_size, 
                                    lr=args.learning_rate, gamma=args.gamma, tau=args.tau,
                                    alpha=args.alpha)

    memory = ReplayMemory(args.capacity)



    solve(args,max_steps,env,agent,memory)
    agent.save_model(args.directory,'final')
