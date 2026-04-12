import numpy as np
import matplotlib.pyplot as plt

#init 4 x4 grid
gamma = 0.9

V = np.zeros((4, 4))

# 0, 0 and 3, 3 are terminal states with reward 0
states = [(i, j) for i in range(4) for j in range(4)]
actions = ['up', 'down', 'left', 'right']

def get_next_state(state, action):
    i, j = state
    if action == 'up':
        return (max(i - 1, 0), j)
    elif action == 'down':
        return (min(i + 1, 3), j)
    elif action == 'left':
        return (i, max(j - 1, 0))
    elif action == 'right':
        return (i, min(j + 1, 3))    

def get_reward(state):
    return -1


# Greedy policy iteration with optimal policy evaluation
def optimal_policy_iteration(max_iterations=1000) -> np.ndarray:
    for _ in range(max_iterations):
        new_V = np.copy(V)
        for state in states:
            if state in [(0, 0), (3, 3)]:
                continue
            action_values = []
            for action in actions:
                next_state = get_next_state(state, action)
                reward = get_reward(next_state)
                action_value = reward + gamma * V[next_state]
                action_values.append(action_value)
            new_V[state] = max(action_values) # optimal policy
        V[:] = new_V

# Non-greedy policy iteration with equal probability for each action
def value_iteration(max_iterations=1000) -> np.ndarray:
    for _ in range(max_iterations):
        new_V = np.copy(V)
        for state in states:
            if state in [(0, 0), (3, 3)]:
                continue
            action_values = []
            for action in actions:
                next_state = get_next_state(state, action)
                reward = get_reward(next_state)
                action_value = reward + gamma * V[next_state]
                action_values.append(action_value)
            new_V[state] = round(sum(action_values) / len(action_values), 10) # equal probability for each action
        V[:] = new_V


if __name__ == "__main__":
    command = input("Enter number of iterations for value iteration (default 1000): ")
    max_iterations = int(command) if command else 1000
    value_iteration(max_iterations)
    print("Final Value Function:")
    print(V)
    command = input("Enter number of iterations for optimal policy iteration (default 10): ")
    max_iterations = int(command) if command else 10
    optimal_policy_iteration(max_iterations)
    print("Optimal Value Function:")
    print(V)
    