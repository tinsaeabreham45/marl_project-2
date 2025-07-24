import numpy as np
from pettingzoo.mpe import simple_spread_v3
import random

class MultiAgentEnv:
    def __init__(self, num_agents=3, max_cycles=100, agent_specializations=None):
        self.env = simple_spread_v3.parallel_env(
            N=num_agents,
            local_ratio=0.5,
            max_cycles=max_cycles,
            # render_mode='human',
            continuous_actions=False
        )
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.agent_specializations = agent_specializations or {}
        self.env.reset()
        self.observation_spaces = {agent: self.env.observation_space(agent) for agent in self.env.agents}
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.env.agents}

    def reset(self):
        observations = self.env.reset()
        # Apply agent specialization (e.g., blind agent)
        for agent, spec in self.agent_specializations.items():
            if spec == 'blind' and agent in observations:
                obs = observations[agent]
                obs = np.zeros_like(obs)  # Blind agent sees nothing
                observations[agent] = obs
        return observations

    def step(self, actions, failed_agents=None):
        # failed_agents: list of agent names to fail this step
        failed_agents = failed_agents or []
        for agent in failed_agents:
            if agent in actions:
                # Set action to random or noop (0)
                actions[agent] = self.action_spaces[agent].sample()
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        # Apply agent specialization (e.g., blind agent)
        for agent, spec in self.agent_specializations.items():
            if spec == 'blind' and agent in observations:
                obs = observations[agent]
                obs = np.zeros_like(obs)
                observations[agent] = obs
        done = any(terminations.values()) or any(truncations.values())
        return observations, rewards, done, infos

    def get_state(self):
        # Global state is concatenation of all agent positions
        return np.concatenate([self.env.unwrapped.world.agents[i].state.p_pos for i in range(self.num_agents)])

    def close(self):
        self.env.close()