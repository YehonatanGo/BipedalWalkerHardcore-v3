import torch.nn as nn


class DuelingDQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, state):
        features = self.shared_layer(state)
        values = self.value_layer(features)
        advantages = self.advantage_layer(features)
        qvals = values + (advantages - advantages.mean())
        return qvals
