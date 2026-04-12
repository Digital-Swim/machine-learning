
# All states and values for the tic tac toe game
# Author: Ranjit Nagi
# The state is represented as a string of 9 characters, where each character can be 'X', 'O', or ' ' (space) representing an empty cell. The value of each state is stored in a dictionary, where the key is the state string and the value is the expected reward for that state.
import random

learning_rate = 0.1
V = {}

def lookup(state):
    """Returns the value of the given state."""
    return V.get(state, 0.5)  # Default value is 0.5 for unseen states

def update(state, value):
    """Updates the value of the given state."""
    V[state] = value

# Return all the states for this episode, and the final reward for this episode
# to update the values of the states based on the reward received at the end of the episode. The reward can be +1 for a win, -1 for a loss, and 0 for a draw.
def run_episode():
    """Simulates an episode of the tic tac toe game and updates the values of the states."""
    # This function should implement the logic to simulate a game, update the state values based on the outcome, and return the final reward.
    board = [' '] * 9  # Initialize an empty board
    state = ''.join(board)
    states_visited = []  # List to keep track of states visited during the episode
    # Simulate the game until it ends
    while True:
        # Player X's turn (agent)
        action = choose_action(state)  # Implement a function to choose an action based on the current state
        board[action] = 'X'
        state = ''.join(board)
        states_visited.append(state)  # Add the current state to the list of visited states
        
        if check_winner(board, 'X'):
            update(state, 1)  # Agent wins
            return states_visited, 1
        elif check_draw(board):
            update(state, -1)  # Draw
            return states_visited, -1
        
        # Player O's turn (opponent)
        opponent_action = choose_opponent_action(state)  # Implement a function to choose an opponent action
        board[opponent_action] = 'O'
        state = ''.join(board)
        
        if check_winner(board, 'O'):
            update(state, -1)  # Opponent wins
            return states_visited, -1
        elif check_draw(board):
            update(state, -1)  # Draw
            return states_visited, -1
    
def check_winner(board, player):
    """Checks if the given player has won the game."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

def check_draw(board):
    """Checks if the game is a draw."""
    return ' ' not in board


# use randon in 10 percent of the time to choose a random action, and the best action based on the current values in 90
# percent of the time.
def choose_action(state):
    """Chooses an action for the agent based on the current state."""
    if random.random() < 0.1:  # 10% chance to choose a random action
        available_actions = [i for i in range(9) if state[i] == ' ']
        return random.choice(available_actions)
    else:  # 90% chance to choose the best action based on current values
        best_value = -float('inf')
        best_action = None
        for i in range(9):
            if state[i] == ' ':
                next_state = state[:i] + 'X' + state[i+1:]  # Simulate the action
                value = lookup(next_state)
                if value > best_value:
                    best_value = value
                    best_action = i
        return best_action

def choose_opponent_action(state):
    """Chooses an action for the opponent based on the current state."""
    # Implement a strategy for the opponent, such as random selection or a simple heuristic.
    available_actions = [i for i in range(9) if state[i] == ' ']
    return random.choice(available_actions)

def main():
    """Main function to run multiple episodes and train the agent."""
    for episode in range(100000):  # Run 10,000 episodes
        states_visited, reward = run_episode()
        # Update the values of the states visited during the episode based on the final reward
        # final reward received at the end of the episode. The reward can be +1 for a win, -1 for a loss, and 0 for a draw.
        # The update rule can be: new_value = old_value + learning_rate * (reward - old_value), where learning_rate is a parameter that controls how much the values are updated based on the reward.
        # we dont udate the winning or losing state, as they are already updated in the run_episode function, we only update the states that were visited before the final state.
        for state in states_visited[:-1]:  # Exclude the final state which is already updated
            old_value = lookup(state)
            new_value = old_value + learning_rate * (reward - old_value)  # Using a learning rate of 0.1
            update(state, new_value)
        
# store the values in a file
def save_values(filename):
    """Saves the values of the states to a file."""
    with open(filename, 'w') as f:
        for state, value in V.items():
            f.write(f"{state}:{value}\n")

if __name__ == "__main__":
    main()
    save_values("values.txt")
