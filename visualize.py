import matplotlib.pyplot as plt
import numpy as np

def plot_results(episode_rewards, losses, agent_distances=None, filename="training_results.png", agent_rewards=None):
    plt.figure(figsize=(15, 5))

    # Plot episode rewards
    plt.subplot(1, 3 if agent_distances is not None else 2, 1)
    plt.plot(episode_rewards, label="Episode Reward")
    plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), label="Moving Avg (100)")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # Plot losses
    plt.subplot(1, 3 if agent_distances is not None else 2, 2)
    if len(losses) > 0:
        plt.plot(losses, label="Loss")
        if len(losses) >= 1000:
            plt.plot(np.convolve(losses, np.ones(1000)/1000, mode='valid'), label="Moving Avg (1000)")
        plt.title("Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
    else:
        plt.title("Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.text(0.5, 0.5, "No loss data", ha='center')

    # Plot agent distances if provided
    if agent_distances is not None:
        plt.subplot(1, 3, 3)
        plt.plot(agent_distances, label="Avg Agent Distance")
        plt.plot(np.convolve(agent_distances, np.ones(100)/100, mode='valid'), label="Moving Avg (100)")
        plt.title("Agent Coordination (Avg Distance)")
        plt.xlabel("Episode")
        plt.ylabel("Distance")
        plt.legend()

    # Plot per-agent rewards if provided
    if agent_rewards is not None:
        plt.figure(figsize=(10, 5))
        for agent, rewards in agent_rewards.items():
            plt.plot(rewards, label=agent)
        plt.title("Per-Agent Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig("per_agent_rewards.png")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()