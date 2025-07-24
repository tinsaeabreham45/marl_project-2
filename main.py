from environment import MultiAgentEnv
from qmix import QMIX
from train import train_qmix
from visualize import plot_results
import numpy as np
import random

def compute_avg_distance(env):
    positions = [agent.state.p_pos for agent in env.env.unwrapped.world.agents]
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(np.linalg.norm(positions[i] - positions[j]))
    return np.mean(distances) if distances else 0

def main():
    # Make agent_1 blind
    env = MultiAgentEnv(num_agents=3, max_cycles=100, agent_specializations={'agent_1': 'blind'})
    
    obs_dim = env.observation_spaces['agent_0'].shape[0]
    act_dim = env.action_spaces['agent_0'].n
    state_dim = env.num_agents * 2  # Position coordinates for each agent
    qmix = QMIX(num_agents=3, state_dim=state_dim, obs_dim=obs_dim, act_dim=act_dim)

    # Training phase
    episode_rewards, losses = train_qmix(env, qmix, num_episodes=500)

    # Robustness/evaluation phase
    num_episodes = 1000
    fail_prob = 0.2
    failed_agent_log = []
    agent_rewards = {f'agent_{i}': [] for i in range(env.num_agents)}
    episode_rewards_eval = []
    agent_distances = []
    for episode in range(num_episodes):
        # obs = env.reset()
        done = False
        failed_agents = []
        if random.random() < fail_prob:
            failed_agent = f'agent_{random.randint(0, env.num_agents-1)}'
            failed_agents = [failed_agent]
        failed_agent_log.append(failed_agents)
        episode_reward = 0
        per_agent_reward = {f'agent_{i}': 0.0 for i in range(env.num_agents)}
        messages = [0.0 for _ in range(env.num_agents)]
        while not done:
            actions, messages = qmix.select_actions(obs, epsilon=0.05, episode_fingerprint=episode, messages=messages)
            obs, rewards, done, _ = env.step(actions, failed_agents=failed_agents)
            for i in range(env.num_agents):
                per_agent_reward[f'agent_{i}'] += rewards[f'agent_{i}']
            episode_reward += sum(rewards.values()) / env.num_agents    # average reward per agent
            agent_distances.append(compute_avg_distance(env))
        for i in range(env.num_agents):
            agent_rewards[f'agent_{i}'].append(per_agent_reward[f'agent_{i}'])
        episode_rewards_eval.append(episode_reward)
        print(f"Episode {episode}, Failed agents: {failed_agents}, Rewards: {[per_agent_reward[f'agent_{i}'] for i in range(env.num_agents)]}")

    # Visualize results
    plot_results(episode_rewards, losses, agent_distances, filename="qmix_training_results.png", agent_rewards=agent_rewards)

    env.close()

if __name__ == "__main__":
    main()