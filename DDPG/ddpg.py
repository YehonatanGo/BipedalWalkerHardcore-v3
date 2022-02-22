import torch
from DDPG.Models import *
from DDPG.Memory import *
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_size, action_size, seed):
        """
        :param state_size: number of variables in each state
        :param action_size: number of variables in each action
        :param seed:
        """
        self.state_size = state_size
        self.action_size = action_size
        self.timestep = 0
        self.batch_size = 64
        self.buffer_size = 1000000
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.gamma = .99
        self.tau = 0.001
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996
        self.should_be_updated = 50

        # ---------networks initialization--------#
        # -----actor networks-----#
        self.actor_online = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_online.parameters(), lr=self.actor_lr)

        # -----critic networks-----#
        self.critic_online = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_online.parameters(), lr=self.critic_lr)

        # -----initialize memory-----#
        self.memory = Memory(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=seed)

    def step(self, state, action, reward, next_state, done):
        self.timestep += 1
        # add current experience to the memory
        self.memory.add(state, action, reward, next_state, done)
        # if there's enough experience samples in the memory - learn
        if len(self.memory) > self.batch_size:
            sampled_experiences = self.memory.sample()
            self.learn(sampled_experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ----update actor----#
        predicted_actions = self.actor_online(states)
        actor_loss = -self.critic_online(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----update critic----#
        next_pred_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_pred_actions)
        curr_Q_targets = rewards + (self.gamma * next_Q_targets * (1 - dones))
        # compute loss
        curr_Q_predicted = self.critic_online(states, actions)
        critic_loss = F.mse_loss(curr_Q_predicted, curr_Q_targets)
        # step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        if self.timestep % self.should_be_updated == 0:
            # critic
            for target_param, local_param in zip(self.critic_target.parameters(), self.critic_online.parameters()):
                target_param.data.copy_(local_param.data)
            # actor
            for target_param, local_param in zip(self.actor_target.parameters(), self.actor_online.parameters()):
                target_param.data.copy_(local_param.data)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        # TODO: add noise?

        rnd = random.random()
        if rnd < self.epsilon:
            return np.random.uniform(-1, 1, size=(4,))
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.actor_online.eval()
            with torch.no_grad():
                action = self.actor_online(state).cpu().data.numpy().squeeze()
            self.actor_online.train()
            return action

    def solve(self, env, num_of_episodes=3000):
        rewards = []
        for episode in range(num_of_episodes):
            state = env.reset()
            score = 0
            max_steps = 3000
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                env.render()
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    print(f"Episode: {episode}/{num_of_episodes}, score: {score}", end="\r")
                    break
            rewards.append(score)
            is_solved = np.mean(rewards[-100:])
            if is_solved >= 200:
                print("\n")
                print(f"Enviroment solved in {episode} episodes!")
                break
            if episode % 100 == 0 and episode != 0:
                print(f"Average score in episode {episode} is: {is_solved}")

        return rewards
