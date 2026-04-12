import random
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = [0] * 9

    def reset(self):
        self.board = [0] * 9
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action, player):
        if self.board[action] != 0:
            return self.get_state(), -1, True

        self.board[action] = player

        winner = self.check_winner()
        if winner != 0:
            return self.get_state(), winner, True

        if 0 not in self.board:
            return self.get_state(), 0, True

        return self.get_state(), 0, False

    def check_winner(self):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for w in wins:
            s = sum(self.board[i] for i in w)
            if s == 3: return 1
            if s == -3: return -1
        return 0


Q = defaultdict(lambda: [0]*9)

alpha = 0.1
gamma = 0.9
epsilon = 0.2

env = TicTacToe()

def choose_action(state, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    q_values = Q[state]
    return max(actions, key=lambda a: q_values[a])

def opponent_move(env):
    actions = env.available_actions()
    return random.choice(actions)


for episode in range(50000):
    state = env.reset()
    done = False

    while not done:
        actions = env.available_actions()
        action = choose_action(state, actions)

        next_state, reward, done = env.step(action, 1)

        if not done:
            opp_action = opponent_move(env)
            next_state, reward, done = env.step(opp_action, -1)

            if done:
                reward = -1 if reward == -1 else reward

        best_next = max(Q[next_state]) if not done else 0

        Q[state][action] += alpha * (
            reward + gamma * best_next - Q[state][action]
        )

        state = next_state
        
def print_board(board):
    symbols = {1: "X", -1: "O", 0: " "}
    for i in range(3):
        row = board[i*3:(i+1)*3]
        print(" | ".join(symbols[x] for x in row))
        if i < 2:
            print("--+---+--")


def play():
    state = env.reset()
    done = False

    print("\nYou are O (-1). AI is X (1)\n")

    while not done:

        # AI move
        ai_action = max(env.available_actions(), key=lambda a: Q[state][a])
        state, reward, done = env.step(ai_action, 1)

        print("\nAI move:")
        print_board(state)

        if done:
            break

        # Human move
        print("\nYour turn. Available:", env.available_actions())

        try:
            move = int(input("Enter move (0-8): "))
        except:
            print("Invalid input")
            continue

        if move not in env.available_actions():
            print("Invalid move, try again.")
            continue

        state, reward, done = env.step(move, -1)

        print("\nYour move:")
        print_board(state)

    print("\nGame Over. Result:", reward)


if __name__ == "__main__":
    play()