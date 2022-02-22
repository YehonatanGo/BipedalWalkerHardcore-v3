# used https://github.com/claudeHifly/BipedalWalker-v3/blob/master as template

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network maps states to an action
    """
    def __init__(self, state_size, action_size):
        """
        Build a fully connected neural network
        """
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)

        # self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.torch.tanh(x)


class Critic(nn.Module):
    """
    Critic network maps (state, action) pair to its value
    """
    def __init__(self, state_size, action_size):

        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        """
        approximate Q(state, action)
        """
        state_act = torch.cat([state, action], 1)

        x = F.relu(self.fc1(state_act))
        x = F.relu(self.fc2(x))
        q_val = self.fc3(x)
        return q_val
