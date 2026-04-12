import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)
        self.q_nonstationary = np.zeros(k)  # Start with a high value to encourage exploration
        self.optimal_action = np.argmax(self.q_true)

    def pull(self, action):
        return np.random.normal(self.q_true[action], 1)
    
    def get_optimal_action(self):
        return np.argmax(self.q_true)

    def pull_nonstationary(self):
        self.q_nonstationary += np.random.normal(0, 0.01, self.k)
        return self.q_nonstationary
    
    def get_optimal_action_nonstationary(self):
        return np.argmax(self.q_nonstationary)


# Given epsilon, k , alpha  
class EpsilonGreedyAgentNonStationary:
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])


class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1, initial_value=0):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.full(k, initial_value)
        self.N = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

def run_experiment_nonstationary(epsilon, runs=2000, steps=1000, k=10):
    avg_reward = np.zeros(steps)
    optimal_action_rate = np.zeros(steps)

    for _ in range(runs):
        bandit = Bandit(k)
        agent = EpsilonGreedyAgentNonStationary(k, epsilon)

        rewards = np.zeros(steps)
        optimal_flags = np.zeros(steps)

        for t in range(steps):
            action = agent.select_action()
            reward = bandit.pull_nonstationary()[action]
            agent.update(action, reward)

            rewards[t] = reward
            optimal_flags[t] = (action == bandit.get_optimal_action_nonstationary())

        avg_reward += rewards
        optimal_action_rate += optimal_flags

    avg_reward /= runs
    optimal_action_rate = (optimal_action_rate / runs) * 100

    return avg_reward, optimal_action_rate 

def run_experiment(epsilon, runs=2000, steps=1000, k=10, initial_value=0):
    avg_reward = np.zeros(steps)
    optimal_action_rate = np.zeros(steps)

    for _ in range(runs):
        bandit = Bandit(k)
        agent = EpsilonGreedyAgent(k, epsilon, initial_value)

        rewards = np.zeros(steps)
        optimal_flags = np.zeros(steps)

        for t in range(steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)

            rewards[t] = reward
            optimal_flags[t] = (action == bandit.get_optimal_action())
            

        avg_reward += rewards
        optimal_action_rate += optimal_flags

    avg_reward /= runs
    optimal_action_rate = (optimal_action_rate / runs) * 100

    return avg_reward, optimal_action_rate


def main():
    epsilons = [0, 0.1]
    results = {}

    for eps in epsilons:
        results[eps] = run_experiment(eps, runs=2000, steps=1000, k=10, initial_value=5.0)

    # Plot results
    plt.figure(figsize=(10, 8))

    # Average reward
    plt.subplot(2, 1, 1)
    for eps in epsilons:
        avg_reward, _ = results[eps]
        plt.plot(avg_reward, label=f"ε = {eps}")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()

    # Optimal action %
    plt.subplot(2, 1, 2)
    for eps in epsilons:
        _, opt = results[eps]
        plt.plot(opt, label=f"ε = {eps}")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()