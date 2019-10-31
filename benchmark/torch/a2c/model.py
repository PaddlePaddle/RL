import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import parl


class ActorCritic(parl.Model):
    def __init__(self, act_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=2)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.fc = nn.Linear(7744, 512)

        self.fc_pi = nn.Linear(512, act_dim)
        self.fc_v = nn.Linear(512, 1)

    def policy(self, x, softmax_dim=1):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))

        logits = self.fc_pi(x)
        prob = F.softmax(logits, dim=softmax_dim)

        return prob

    def value(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))
        values = self.fc_v(x)

        return values

    def policy_and_value(self, x, softmax_dim=1):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))

        values = self.fc_v(x)
        logits = self.fc_pi(x)
        prob = F.softmax(logits, dim=softmax_dim)

        return prob, values

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
