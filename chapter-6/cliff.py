# Cliff Walking Example
# Grid of size 4 rows and 12 columns, with the start state at (3, 0) and the goal state at (3, 11).
# The cliff is represented by the states (3, 1) to (3, 10), which give a reward of -100 if the agent steps on them.
# The agenet returns to the start state if it steps on the cliff
# The agent receives a reward of -1 for each step taken, and a reward of 0 for reaching the goal state. 
# The actions available to the agent are up, down, left, and right.

import random
import time
import os

states = [(row, col) for row in range(4) for col in range(12)]
actions = ['up', 'down', 'left', 'right']
terminal_states = [(3, 11)]
start_state = (3, 0)

action_symbols = {
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→'
    
}

# Using Q Learning here 
class CliffWalking:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.grid_size = (4, 12)  # (rows, columns)
        self.start_state = (3, 0)  # (row, column)
        self.goal_state = (3, 11)  # (row, column)

        self.Q = { (state, action): 0.0 for state in states for action in actions}

    def get_next_state(self, state, action):
        if state in terminal_states:
            return state
        row, col = state
        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            col -= 1
        elif action == 'right':
            col += 1
        
        if col in range(1, 11) and row == 3:  # Cliff states
            return (3, 0), -100  # Return to start state and give penalty
        
        if (row, col) == (3, 11):  # Goal state
            return (3, 11), 0
        
        # Ensure the agent stays within the grid boundaries
        row = max(0, min(3, row))
        col = max(0, min(11, col))
        
        return (row, col) , -1  # Normal step with a reward of -1


    # choose action using epsilon-greedy policy, with epsilon = 0.1
    # The agent will choose a random action with probability 0.1 and the action with
    # the highest value with probability 0.9
    def choose_action(self, state):
        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(actions)
        else:
            # Choose the action with the highest value
            q_values = [self.Q[(state, action)] for action in actions]
            max_q = max(q_values)
            max_actions = [action for action in actions if self.Q[(state, action)] == max_q]
            return random.choice(max_actions)
        
        
    def run_episode(self):
        state = start_state
        while state not in terminal_states:
            action = self.choose_action(state)
            next_state, reward = self.get_next_state(state, action)
            # Q Learning update rule: Q(state, action) += alpha * (reward + gamma * max(Q(next_state, a)) - Q(state, action))
            # Using new state and chosing action with max values for the next state to update the Q values for the current state and action
            self.Q[(state, action)] += self.alpha * (reward + self.gamma * max(self.Q[(next_state, a)] for a in actions) - self.Q[(state, action)])
            state = next_state

    def print_policy(self):
        rows, cols = self.grid_size
        
        for r in range(rows):
            row = ""
            for c in range(cols):
                state = (r, c)

                if state == self.goal_state:
                    row += " G  "
                else:
                    best_action = max(actions, key=lambda a: self.Q[(state, a)])
                    row += f" {action_symbols[best_action]}  "
            print(row)


class CliffWalkingSARSA(CliffWalking):

    def run_episode(self):
        state = start_state
        action = self.choose_action(state)
        while state not in terminal_states:
            next_state, reward = self.get_next_state(state, action)
            next_action = self.choose_action(next_state)
            # SARSA update rule: Q(state, action) += alpha * (reward + gamma * Q(next_state, next_action) - Q(state, action))
            self.Q[(state, action)] += self.alpha * (reward + self.gamma * self.Q[(next_state, next_action)] - self.Q[(state, action)])
            state = next_state
            action = next_action
        

if __name__ == "__main__":

    agentSARSA = CliffWalkingSARSA()
    agenetQLearning = CliffWalking()
    
    num_episodes = int(input("Enter the number of episodes to run: "))
    
    for episode in range(num_episodes):
        agentSARSA.run_episode()
    
    for episode in range(num_episodes):
        agenetQLearning.run_episode()
        
    print("Learned Policy: SARSA")
    agentSARSA.print_policy()
    
    print("\nLearned Policy: Q Learning")
    agenetQLearning.print_policy()
