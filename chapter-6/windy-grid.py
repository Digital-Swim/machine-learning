# Windy Grid World example from Sutton and Barto's "Reinforcement Learning: An Introduction" (Chapter 6, Section 6.5)
# Grid size 10 bt 7 (10 columns and 7 rows), 
# Upward wind in columns 3, 4, 5, 8 with strength 1, 1, 1, and 1 respectively, and in columns 6 and 7 with strength 2 and 2 respectively.
# The agent starts at (3, 0) and aims to reach the goal at (3, 7). (row, column)
# Actions: up, down, left, right
# Eplselon greedy policy with epsilon = 0.1, 
# so the agent will choose a random action with probability 0.1 and the action with the highest value with probability 0.9.

import random
import time
import os

# Using SARSA
class WindyGridWorld:
    action_symbols = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→'
        
}
    def __init__(self, alpha=0.5, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.grid_size = (7, 10)  # (rows, columns)
        self.start_state = (3, 0)  # (row, column)
        self.goal_state = (3, 7)   # (row, column)
        self.wind_strengths = {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1}  # column index to wind strength
        self.actions = ['up', 'down', 'left', 'right']
        self.Q = { (state, action): 0.0 for state in [(row, col) for row in range(self.grid_size[0]) for col in range(self.grid_size[1])] for action in self.actions}
    
    def get_next_state(self, state, action):
        row, col = state
        
        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            col -= 1
        elif action == 'right':
            col += 1
        
        # Apply wind effect
        if col in self.wind_strengths:
            row -= self.wind_strengths[col]
        
        # Ensure the agent stays within the grid boundaries
        row = max(0, min(self.grid_size[0] - 1, row))
        col = max(0, min(self.grid_size[1] - 1, col))
        
        return (row, col)

    def get_reward(self, state):
        if state == self.goal_state:
            return 0  # reward for reaching the goal
        else:
            return -1  # penalty for each step taken
        
    
    # choose action using epsilon-greedy policy, with epsilon = 0.1
    # Thge agent will choose a random action with probability 0.1 and
    # the action with the highest Q value with probability 0.9
    # Q value is the expected return for taking an action in a state, and following the policy thereafter
    # use SARSA update rule to update the Q values for each state-action pair in the episode
    # The SARSA update rule is: Q(state, action) += alpha * (reward + gamma * Q(next_state, next_action) - Q(state, action))
    
    def choose_action(self, state):
        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            # Choose the action with the highest Q value for the current state
            max_Q = max(self.Q[(state, action)] for action in self.actions)
            best_actions = [action for action in self.actions if self.Q[(state, action)] == max_Q]
            return random.choice(best_actions)
        
    def update(self, state, action, reward, next_state):
        alpha = self.alpha  # learning rate
        gamma = self.gamma  # discount factor
        #choose next action from next state using epsilon-greedy policy
        next_action = self.choose_action(next_state)
        next_Q = self.Q[(next_state, next_action)]
        self.Q[(state, action)] += alpha * (reward + gamma * next_Q - self.Q[(state, action)])
        
    # using TD(0) learning to update the Q values for each state-action pair in the episode
    def run_episode(self):
        state = self.start_state
        
        while state != self.goal_state:
            action = self.choose_action(state)
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            self.update(state, action, reward, next_state)
            state = next_state

    def run_test_episode(self, start):
        state = start
        path = [state]

        while state != self.goal_state:
            action = max(self.actions, key=lambda a: self.Q[(state, a)])
            state = self.get_next_state(state, action)  # includes wind
            path.append(state)

        return path
    
    def print_path(self, path):
        rows, cols = self.grid_size
        grid = [['.' for _ in range(cols)] for _ in range(rows)]

        for (r, c) in path:
            grid[r][c] = '*'

        gr, gc = self.goal_state
        grid[gr][gc] = 'G'

        for row in grid:
            print(" ".join(row))

    def animate(self, start_state):
        state = start_state
        rows, cols = self.grid_size

        while state != self.goal_state:
            os.system('cls')  # use 'cls' on Windows

            # create empty grid
            grid = [['.' for _ in range(cols)] for _ in range(rows)]

            # mark agent
            r, c = state
            grid[r][c] = 'A'

            # mark goal
            gr, gc = self.goal_state
            grid[gr][gc] = 'G'

            # print grid
            for row in grid:
                print(" ".join(row))

            print(" ".join(['0', '0', '0', '1', '1', '1', '2', '2', '1', '0']))  # wind strengths for each column
            time.sleep(0.3)

            # choose greedy action
            action = max(self.actions, key=lambda a: self.Q[(state, a)])

            # move (IMPORTANT: must include wind)
            state = self.get_next_state(state, action)

        # final state
        os.system('cls')  # use 'cls' on Windows
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        gr, gc = state
        grid[gr][gc] = 'G'

        for row in grid:
            print(" ".join(row))
        
        print("\nReached Goal ✅")
        
    def print_policy(self):
        rows, cols = self.grid_size
        
        for r in range(rows):
            row = ""
            for c in range(cols):
                state = (r, c)

                if state == self.goal_state:
                    row += " G  "
                else:
                    best_action = max(
                        self.actions,
                        key=lambda a: self.Q[(state, a)]
                    )
                    row += f" {self.action_symbols[best_action]}  "
            print(row)
            
if __name__ == "__main__":
    agent = WindyGridWorld()

    num_episodes = 200000
    for episode in range(num_episodes):
        agent.run_episode()
    
    print("Learned Policy:")
    agent.print_policy()
    #agent.print_path(agent.run_test_episode(start=(5, 0)))
    agent.animate(start_state=(6, 0))
        
        