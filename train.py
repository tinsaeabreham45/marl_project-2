import numpy as np
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, state_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.observations = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.fingerprints = deque(maxlen=capacity)  # Store episode number for each agent

    def add(self, state, actions, rewards, next_state, done, obs, next_obs, fingerprint):
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_state)
        self.dones.append({f'agent_{i}': done for i in range(self.num_agents)})
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.fingerprints.append(fingerprint)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size)
        states = np.array([self.states[i] for i in indices])
        actions = {f'agent_{i}': np.array([self.actions[j][f'agent_{i}'] for j in indices]) for i in range(self.num_agents)}
        rewards = {f'agent_{i}': np.array([self.rewards[j][f'agent_{i}'] for j in indices]) for i in range(self.num_agents)}
        next_states = np.array([self.next_states[i] for i in indices])
        dones = {f'agent_{i}': np.array([self.dones[j][f'agent_{i}'] for j in indices]) for i in range(self.num_agents)}
        observations = {f'agent_{i}': np.array([self.observations[j][f'agent_{i}'] for j in indices]) for i in range(self.num_agents)}
        next_observations = {f'agent_{i}': np.array([self.next_observations[j][f'agent_{i}'] for j in indices]) for i in range(self.num_agents)}
        fingerprints = {f'agent_{i}': np.array([self.fingerprints[j][f'agent_{i}'] for j in indices]) for i in range(self.num_agents)}
        return states, actions, rewards, next_states, dones, observations, next_observations, fingerprints

    def __len__(self):
        return len(self.states)

def train_qmix(env, qmix, num_episodes=1000, batch_size=32, buffer_capacity=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    buffer = ReplayBuffer(buffer_capacity, env.num_agents, env.observation_spaces['agent_0'].shape[0], env.num_agents * 2)
    epsilon = epsilon_start
    episode_rewards = []
    losses = []

    for episode in range(num_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        episode_reward = 0
        done = False
        while not done:
            actions, _ = qmix.select_actions(obs, epsilon, episode_fingerprint=episode)
            result = env.step(actions)
            next_obs = result[0] if isinstance(result, tuple) else result
            rewards = result[1] if isinstance(result, tuple) else result
            done = result[2] if isinstance(result, tuple) else any(result[3].values()) or any(result[4].values())
            state = env.get_state()
            next_state = env.get_state()
            # Store fingerprint (episode number) for each agent
            fingerprint = {f'agent_{i}': episode for i in range(env.num_agents)}
            buffer.add(state, actions, rewards, next_state, done, obs, next_obs, fingerprint)
            episode_reward += sum(rewards.values()) / env.num_agents
            obs = next_obs

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = qmix.train(batch, episode_fingerprint=episode)
                losses.append(loss)

        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % 10 == 0:
            qmix.update_target_networks()
            print(f"Episode {episode}, Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}, Epsilon: {epsilon:.2f}")

    return episode_rewards, losses