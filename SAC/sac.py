import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import *
from torch.optim import Adam
from collections import deque
from Memory import ReplayMemory
from utils import soft_update, hard_update
from models import GaussianPolicy, QNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SACAgent(object):
    def __init__(self, state_size, action_space, hidden_size):
        self.lr =  0.00008
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.buffer_size = 1000000
        self.batch_size = 256
        self.start_steps = 10000
        self.action_size = action_space.shape[0]
        self.state_size = state_size

        #-----critics networks-----#
        self.critic = QNetwork(self.state_size, self.action_size, hidden_size).to(device=device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target = QNetwork(state_size, self.action_size, hidden_size).to(device)
        hard_update(self.critic_target, self.critic)
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        #-----policy networks-----#
        self.policy = GaussianPolicy(self.state_size, self.action_size, hidden_size, action_space).to(device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        #-----initialize memory-----#
        self.memory = ReplayMemory(self. buffer_size)

    def choose_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)
    
    def solve(self, env, num_of_episodes=3000):
        total_numsteps = 0
        scores_deque = deque(maxlen=100)
        scores_array = []
        avg_scores_array = [] 
        episodes_array = []
        max_steps = 2000
        is_finished = False

        for episode in range(num_of_episodes): 
            score = 0
            episode_steps = 0
            done = False
            state = env.reset()
            for _ in range(max_steps):    
                if self.start_steps > total_numsteps:
                    action = env.action_space.sample()  # Sample random action
                else:
                    action = self.choose_action(state)  # Sample action from policy

                if len(self.memory) > self.batch_size:
                    # Update parameters of all the networks
                    self.update_parameters()

                next_state, reward, done, _ = env.step(action) # Step
                if reward == -100:
                    reward = -1
                # env.render()
                episode_steps += 1
                total_numsteps += 1
                score += reward
                mask = 1 if episode_steps == 2000 else float(not done)
                self.memory.push(state, action, reward, next_state, mask) # Append transition to memory
                state = next_state
                if done:
                    scores_deque.append(score)
                    scores_array.append(score)        
                    avg_score = np.mean(scores_deque)
                    avg_scores_array.append(avg_score)
                    episodes_array.append(episode)
                    print(f"Episode: {episode}, Score: {score}, Avg.Score: {avg_score}", end="\r")
                    break
            if avg_score >= 300:
                is_finished = True
                break
            if episode >= 2900:
                is_finished = self.test(env, max_steps, episode, avg_score)  
            if is_finished == True:
                break
        
        plt.title("SAC")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.plot(episodes_array,scores_array,avg_scores_array)
        plt.savefig("SAC.jpg")
        plt.show()
        return scores_array, avg_scores_array 



    def test(self, env , max_steps, episode, avg_score):
        print("\n")
        print("################################################################")
        print("Current score is 300! let's try 100 episodes to see if we are done!")
        print("################################################################")
        rewards_over_100 = []
        is_finished = False

        for e in range(100):
            state = env.reset()
            temp_score = 0
            for _ in range(max_steps):
                action = self.choose_action(state, eval=True)
                next_state, reward, done, _ = env.step(action) # Step
                # env.render()
                state = next_state
                temp_score += reward
                if done:
                    print(f"Episode: {e}/100, score: {temp_score}", end="\r")
                    break
            rewards_over_100.append(temp_score)
        
        result = np.mean(rewards_over_100)
        if result >= 300:
            print("\n")
            print(f"Enviroment solved in {episode} episodes!")
            print('Solved environment with Avg Score:  ', avg_score)
            is_finished = True
        else:
            print(f"Enviroment not solved yet! Average score over 100: {result}\n") 

        return is_finished   

    # Save model parameters
    def save_model(self, directory = 'models', suffix = '1'):
        print('Saving models to {}{}'.format(directory, suffix))
        if not os.path.exists('%s'%(directory)):
            os.makedirs('%s'%(directory))
        torch.save(self.policy.state_dict(), '%s/actor_%s.pth' % (directory, suffix))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pth' % (directory, suffix))

    # Load model parameters
    def load_model(self, directory = 'models', suffix = '1'):
        print('Loading models from {}{}'.format(directory, suffix))
        if directory is not None:
            self.policy.load_state_dict(torch.load('%s/actor_%s.pth' % (directory, suffix), map_location=torch.device('cpu')))
            self.critic.load_state_dict(torch.load('%s/critic_%s.pth' % (directory, suffix), map_location=torch.device('cpu')))


