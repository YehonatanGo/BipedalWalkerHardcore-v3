# used the help of this blog tutorial: https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
# dueling networks paper: https://arxiv.org/pdf/1511.06581.pdf

from D3QN.network import *
from D3QN.memory import *
from D3QN.discretization import *
import torch.optim as optim
import torch
import torch.nn.functional as F


class D3QNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.timestep = 0
        self.batch_size = 256
        self.buffer_size = 1000000
        self.lr = 0.00008
        self.gamma = .99
        self.tau = 0.01
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996

        # -----network initialization----- #
        self.Q = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        # -----memory initialization----- #
        self.memory = Memory(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=seed)

        def step(self, state, action, reward, next_state, done):
            self.timestep += 1
            self.memory.add(state, action, reward, next_state, done)
            if len(self.memory) > self.batch_size:
                sampled_experinces = self.memory.sample()
                self.learn(sampled_experinces)

    def step(self, state, action, reward, next_state, done):
        self.timestep += 1
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            sampled_experinces = self.memory.sample()
            self.learn(sampled_experinces)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        dones = dones.squeeze(1)
        rewards = rewards.squeeze(1)

        # prediction
        curr_q_vals = self.Q.forward(states).gather(1, actions).squeeze(1)

        # target
        next_q_vals = self.Q.forward(next_states).squeeze(1)
        max_next_Q = torch.max(next_q_vals, 1)[0]
        target_Q = rewards + (self.gamma * max_next_Q * (1 - dones))

        loss = F.mse_loss(curr_q_vals, target_Q)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ----------------------- decay epsilon ------------------------ #
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        """
        Choose the action
        """
        rnd = random.random()
        if rnd <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.Q.eval()
            with torch.no_grad():
                action_values = self.Q(state)
            self.Q.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def checkpoint(self, filename):
        torch.save(self.Q.state_dict(), filename)

    def solve(self, env, num_of_episodes=3000):
        rewards = []
        time_steps = 0
        state = env.reset()
        done = False
        for episode in range(num_of_episodes):
            state = env.reset()
            score = 0
            max_steps = 2000
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(discrete_actions[action])
                # env.render()
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    print(f"Episode: {episode}/{num_of_episodes}, score: {score}", end="\r")
                    break
            rewards.append(score)
            mean_100_eps = np.mean(rewards[-100:])
            print(f"Episode: {episode}, score: {score:.8}, avg last 100 episodes: {mean_100_eps:.8}")
            if mean_100_eps >= 200:
                print("\n")
                print(f"Enviroment solved in {episode} episodes!")
                break
            if episode % 100 == 0 and episode != 0:
                print(f"Average score in episode {episode} is: {mean_100_eps}")

        return rewards
