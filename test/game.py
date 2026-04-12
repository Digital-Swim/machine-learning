# Lets create a simple tic tac toe game, using the agent that has learned to play the game using reinforcement learning. The agent will play against a random opponent, and we will update the values of the states based on the outcome of each episode.
# Learned value table is in values.txt, which is a dictionary of state-value pairs. The state is represented as a string of 9 characters, where each character can be 'X', 'O', or ' ' (empty). The value is a float between 0 and 1, representing the estimated value of that state for the agent.


import random


V = {}  # Initialize the value table as an empty dictionary

# load the value table from values.txt format is 'XXX OX OO:1' where the first part is the state and the second part is the value. We will use this value table to make decisions during the game, and we will also update it based on the outcomes of the games played. The agent will choose actions based on the values of the states, and we will implement a simple epsilon-greedy strategy for action selection.
def load_values(filename):
    """Loads the value table from a file."""
    with open(filename, 'r') as f:
        for line in f:
            state, value = line.strip().split(':')
            V[state] = float(value)
    
def get_value(state):
    """Returns the value of the given state."""
    return V.get(state, 0.5)  # Default value is 0.5 for unseen states


# Lets create the game with console input and output, where the user can play against the agent. The user will be 'O' and the agent will be 'X'. The game will display the board after each move, and it will also display the outcome of the game (win, lose, draw) at the end of each episode. The user can choose to play multiple episodes, and we will update the value table based on the outcomes of the games played.
def display_board(board):
    """Displays the current state of the board."""
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")
    print("")
    print("")
    print("")
    
    
def get_user_action(board):
    """Prompts the user to enter their action."""
    while True:
        try:
            action = int(input("Enter your move (0-8): "))
            if action in range(9) and board[action] == ' ':
                return action
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Please enter a valid number between 0 and 8.")

def main():
    load_values('values.txt')  # Load the value table from the file
    while True:
        board = [' '] * 9  # Initialize an empty board
        state = ''.join(board)
        display_board(board)
        
        while True:
            # User's turn
            user_action = get_user_action(board)
            board[user_action] = 'O'
            state = ''.join(board)
            display_board(board)
            
            if check_winner(board, 'O'):
                print("You win!")
                break
            elif check_draw(board):
                print("It's a draw!")
                break
            
            # Agent's turn
            agent_action = choose_agent_action(state)  # Implement a function to choose an action based on the current state
            board[agent_action] = 'X'
            state = ''.join(board)
            display_board(board)
            
            if check_winner(board, 'X'):
                print("Agent wins!")
                break
            elif check_draw(board):
                print("It's a draw!")
                break
        
        play_again = input("Do you want to play again? (y/n): ")
        if play_again.lower() != 'y':
            break

def choose_agent_action(state):
    #choose action from learned values, if there are multiple actions with the same value, choose one of them randomly
    best_value = -float('inf')
    best_actions = []
    for i in range(9):
        if state[i] == ' ':
            next_state = state[:i] + 'X' + state[i+1:]  # Simulate the action
            value = get_value(next_state)
            if value > best_value:
                best_value = value
                best_actions = [i]
            elif value == best_value:
                best_actions.append(i)
    return random.choice(best_actions)  # Choose randomly among the best actions


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



if __name__ == "__main__":
    main()