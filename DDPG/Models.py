# used https://github.com/claudeHifly/BipedalWalker-v3/blob/master as template

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network maps states to an action
    """
    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, action_size)

        self.bn1 = nn.BatchNorm1d(600)
        self.bn2 = nn.BatchNorm1d(300)


        self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return F.torch.tanh(x)


class Critic(nn.Module):
    """
    Critic network maps (state, action) pair to its value
    """
    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, 1)

        self.action_fc = nn.Linear(action_size, 1)

        self.bn1 = nn.BatchNorm1d(600)

        self.fc2.weight.data.uniform_(-1.5e-3, 1.5e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x_state, action):
        x_state = F.relu(self.bn1(self.fc1(x_state)))
        x_state = self.fc2(x_state)
        action = action.type(torch.FloatTensor)
        x_action = self.action_fc(action)
        x = F.relu(torch.add(x_state, x_action))
        return self.fc3(x)

