import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork


class soft_actor_critic_agent(object):
    def __init__(self, num_inputs, action_space, device, hidden_size, lr, gamma, tau, alpha):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.device = device 

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], \
                                         hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

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

