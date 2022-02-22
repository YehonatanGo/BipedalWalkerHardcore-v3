import sys

from TD3.models import *
from TD3.Memory import *
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent:
    def __init__(self, state_size, action_size, seed, env):
        self.state_size = state_size
        self.action_size = action_size
        self.timestep = 0
        self.batch_size = 64
        self.buffer_size = 1000000
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.gamma = .99
        self.tau = 0.001
        self.start_steps = 10000
        self.noise_clip = 0.5
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # ---------networks initialization--------#
        # -----actor networks-----#
        self.actor_online = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_online.parameters(), lr=self.actor_lr)

        # -----critics networks-----#
        # critic 1 - Q1
        self.critic1_online = Critic(state_size, action_size).to(device)
        self.critic1_target = Critic(state_size, action_size).to(device)
        self.critic1_target.load_state_dict(self.critic1_online.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1_online.parameters(), lr=self.critic_lr)

        # critic 2 - Q2
        self.critic2_online = Critic(state_size, action_size).to(device)
        self.critic2_target = Critic(state_size, action_size).to(device)
        self.critic2_target.load_state_dict(self.critic2_online.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2_online.parameters(), lr=self.critic_lr)

        # -----initialize memory-----#
        self.memory = Memory(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=seed)

    def step(self, state, action, reward, next_state, done):
        """
        add an experience to the memory, and make a learning step if there's enough samples in it
        """
        self.timestep += 1
        # add current experience to the memory
        self.memory.add(state, action, reward, next_state, done)
        # if there's enough experience samples in the memory - learn
        if len(self.memory) > self.batch_size:
            sampled_experiences = self.memory.sample()
            self.learn(sampled_experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        #------update critic------#

        # compute target actions
        noise = np.random.normal(0, 0.1, size=self.action_size)
        noise = noise.clip(-self.noise_clip, self.noise_clip)
        target_actions = self.actor_target(next_states)

        # compute target Q values
        target_Q1 = self.critic1_target(next_states, target_actions)
        target_Q2 = self.critic2_target(next_states, target_actions)
        target_Q = rewards + (self.gamma * torch.min(target_Q1,target_Q2) * (1 - dones))

        # current Q prediction
        predicted_Q1 = self.critic1_online(states, actions)
        predicted_Q2 = self.critic2_online(states, actions)

        # compute loss and update critic networks
        critic_loss1 = F.mse_loss(predicted_Q1, target_Q)
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward(retain_graph=True)
        self.critic1_optimizer.step()

        critic_loss2 = F.mse_loss(predicted_Q2, target_Q)
        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()


        # delayed policy updates
        #-----update acotr-----#
        if self.timestep % 2 == 0:
            # Compute actor loss
            actor_loss = -self.critic1_online(states, self.actor_online(states)).mean()
            # update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks

            # update actor target
            for param, target_param in zip(self.actor_online.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # update critic targets
            for param, target_param in zip(self.critic1_online.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1_online.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def act(self, state, noise=0.1):
        """
        choose action according to given state, using online actor
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_online.eval()
        with torch.no_grad():
            action = self.actor_online(state).cpu().data.numpy().squeeze()
        # add noise
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.action_size))
        self.actor_online.train()
        # clip action values to valid action values
        return action.clip(self.action_low, self.action_high)

    def solve(self, env, num_of_episodes=3000):
        rewards = []

        # For a fixed number of steps at the beginning (set with the start_steps keyword argument),
        # the agent takes actions which are sampled from a uniform random distribution over valid actions.
        # After that, it returns to normal TD3 exploration.

        time_steps = 0
        state = env.reset()
        done = False
        while time_steps < self.start_steps:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            self.memory.add(state, action, reward, next_state, done)

            state = next_state
            time_steps += 1

            if done:
                state = env.reset()
                done = False

            print("\rPopulating Buffer {}/{}.".format(time_steps, self.start_steps), end="")
            sys.stdout.flush()


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
            mean_100_eps = np.mean(rewards[-100:])
            print(f"Episode: {episode}, score: {score}, avg last 100 episodes: {mean_100_eps}")
            if mean_100_eps >= 200:
                print("\n")
                print(f"Enviroment solved in {episode} episodes!")
                break
            if episode % 100 == 0 and episode != 0:
                print(f"Average score in episode {episode} is: {mean_100_eps}")

        return rewards



