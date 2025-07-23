import torch
import torch.nn as nn
import numpy as np

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, comm_dim=1, fingerprint_dim=1):
        super(AgentNetwork, self).__init__()
        self.input_dim = obs_dim + comm_dim + fingerprint_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs, message=None, fingerprint=None):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size = obs.shape[0]
        inputs = [obs]
        # Message
        if message is not None:
            if not isinstance(message, torch.Tensor):
                message = torch.FloatTensor(np.atleast_1d(message))
            if message.dim() == 1:
                message = message.unsqueeze(1)
            if message.shape[0] != batch_size:
                message = message.expand(batch_size, -1)
        else:
            message = torch.zeros(batch_size, 1)
        inputs.append(message)
        # Fingerprint
        if fingerprint is not None:
            if not isinstance(fingerprint, torch.Tensor):
                fingerprint = torch.FloatTensor(np.atleast_1d(fingerprint))
            if fingerprint.dim() == 1:
                fingerprint = fingerprint.unsqueeze(1)
            if fingerprint.shape[0] != batch_size:
                fingerprint = fingerprint.expand(batch_size, -1)
        else:
            fingerprint = torch.zeros(batch_size, 1)
        inputs.append(fingerprint)
        x = torch.cat(inputs, dim=-1)
        out = self.network(x)
        if squeeze_output:
            out = out.squeeze(0)
        return out