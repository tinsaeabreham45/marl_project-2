import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent import AgentNetwork

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=64):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hypernetworks for generating weights
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * num_agents)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, q_values, state):
        # q_values: (batch_size, num_agents, 1)
        # state: (batch_size, state_dim)
        batch_size = q_values.size(0)
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.hidden_dim, self.num_agents)
        w1 = w1.transpose(1, 2)  # (batch_size, num_agents, hidden_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_dim)
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Mixing network: monotonic combination
        hidden = F.elu(torch.bmm(q_values.transpose(1, 2), w1) + b1)  # (batch_size, 1, hidden_dim)
        q_tot = torch.bmm(hidden, w2) + b2  # (batch_size, 1, 1)
        return q_tot.squeeze(-1)  # (batch_size, 1)

class QMIX:
    def __init__(self, num_agents, state_dim, obs_dim, act_dim, hidden_dim=64, lr=0.001):
        self.num_agents = num_agents
        self.agents = [AgentNetwork(obs_dim, act_dim, hidden_dim, comm_dim=1, fingerprint_dim=1) for _ in range(num_agents)]
        self.mixing_net = MixingNetwork(num_agents, state_dim, hidden_dim)
        self.target_agents = [AgentNetwork(obs_dim, act_dim, hidden_dim, comm_dim=1, fingerprint_dim=1) for _ in range(num_agents)]
        self.target_mixing_net = MixingNetwork(num_agents, state_dim, hidden_dim)
        self.update_target_networks()

        self.optimizer = torch.optim.Adam(
            list(self.mixing_net.parameters()) + [p for agent in self.agents for p in agent.parameters()],
            lr=lr
        )
        self.gamma = 0.99

    def update_target_networks(self):
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def select_actions(self, observations, epsilon, episode_fingerprint=0, messages=None):
        actions = {}
        if not isinstance(observations, dict):
            observations = observations[0] if isinstance(observations, tuple) else observations
        # Communication: pass message from previous agent 
        if messages is None:
            messages = [0.0 for _ in range(self.num_agents)]
        new_messages = []
        for i, agent in enumerate(self.agents):
            obs = observations[f'agent_{i}']
            msg = messages[i-1] if i > 0 else messages[-1]  # previous agent's message
            fingerprint = float(episode_fingerprint)
            if not isinstance(obs, torch.Tensor):
                obs = torch.FloatTensor(obs)
            q_values = agent(obs, message=msg, fingerprint=fingerprint)
            if np.random.random() < epsilon:
                action = np.random.randint(q_values.shape[-1])
            else:
                action = torch.argmax(q_values).item()
            actions[f'agent_{i}'] = action
            # For next step, message could be the max Q-value (or action, or any scalar)
            new_messages.append(float(q_values.max().item()))
        return actions, new_messages

    def train(self, batch, episode_fingerprint=0):
        states, actions, rewards, next_states, dones, observations, next_observations, fingerprints = batch
        states = torch.FloatTensor(states)  # (batch_size, state_dim)
        actions = torch.LongTensor(np.array([actions[f'agent_{i}'] for i in range(self.num_agents)]))  # (num_agents, batch_size)
        rewards = torch.FloatTensor(np.array([rewards[f'agent_{i}'] for i in range(self.num_agents)]))  # (num_agents, batch_size)
        next_states = torch.FloatTensor(next_states)  # (batch_size, state_dim)
        dones = torch.FloatTensor(np.array([dones[f'agent_{i}'] for i in range(self.num_agents)]))  # (num_agents, batch_size)
        # Communication: use zeros for messages in training (could be improved)
        messages = [torch.zeros(len(observations[f'agent_{i}'])) for i in range(self.num_agents)]
        fingerprints_tensor = [torch.FloatTensor(fingerprints[f'agent_{i}']) for i in range(self.num_agents)]
        observations = [torch.FloatTensor(observations[f'agent_{i}']) for i in range(self.num_agents)]  # List of (batch_size, obs_dim)
        next_observations = [torch.FloatTensor(next_observations[f'agent_{i}']) for i in range(self.num_agents)]  # List of (batch_size, obs_dim)

        # Compute current Q-values
        q_values = [agent(obs, message=torch.zeros(obs.shape[0]), fingerprint=fingerprints_tensor[i]) for i, (agent, obs) in enumerate(zip(self.agents, observations))]  # List of (batch_size, act_dim)
        q_values = torch.stack(q_values, dim=1)  # (batch_size, num_agents, act_dim)
        chosen_action_qvals = q_values.gather(2, actions.t().unsqueeze(-1)).squeeze(-1)  # (batch_size, num_agents)
        chosen_action_qvals = chosen_action_qvals.unsqueeze(-1)  # (batch_size, num_agents, 1)
        q_tot = self.mixing_net(chosen_action_qvals, states)  # (batch_size, 1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = [target_agent(obs, message=torch.zeros(obs.shape[0]), fingerprint=fingerprints_tensor[i]) for i, (target_agent, obs) in enumerate(zip(self.target_agents, next_observations))]
            next_q_values = torch.stack(next_q_values, dim=1)  # (batch_size, num_agents, act_dim)
            next_max_q = next_q_values.max(dim=2)[0]  # (batch_size, num_agents)
            next_max_q = next_max_q.unsqueeze(-1)  # (batch_size, num_agents, 1)
            target_q_tot = self.target_mixing_net(next_max_q, next_states)  # (batch_size, 1)
            target = rewards.mean(dim=0, keepdim=True).t() + self.gamma * (1 - dones.mean(dim=0, keepdim=True).t()) * target_q_tot

        # Compute loss
        loss = F.mse_loss(q_tot, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()