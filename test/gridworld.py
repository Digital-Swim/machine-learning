import random
import matplotlib.pyplot as plt
import numpy as np

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        self.obstacles = {(3,3), (3,4), (4,3)}
        self.traps = {(5,5), (6,2)}

        self.shortcut = {(2,2): (7,7)}
        self.shortcut_reward = 2

    def reset(self):
        return self.start


    def render(self, q_table=None, state=None):
        grid = np.zeros((self.size, self.size))
        fig, ax = plt.subplots(figsize=(6,6))

        for x in range(self.size):
            for y in range(self.size):
                cell = (x, y)

                if cell in self.obstacles:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))
                elif cell in self.traps:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='red'))
                elif cell == self.goal:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='green'))
                else:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False))

                if q_table and cell in q_table:
                    q = q_table[cell]

                    if "UP" in q:
                        ax.text(x+0.5, y+0.8, f"{q['UP']:.1f}", ha='center', fontsize=6)
                    if "DOWN" in q:
                        ax.text(x+0.5, y+0.2, f"{q['DOWN']:.1f}", ha='center', fontsize=6)
                    if "LEFT" in q:
                        ax.text(x+0.2, y+0.5, f"{q['LEFT']:.1f}", va='center', fontsize=6)
                    if "RIGHT" in q:
                        ax.text(x+0.8, y+0.5, f"{q['RIGHT']:.1f}", va='center', fontsize=6)

        if state:
            x, y = state
            ax.add_patch(plt.Circle((x+0.5, y+0.5), 0.3, color='blue'))

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid()

        plt.gca().invert_yaxis()
        plt.show()
        
    def in_bounds(self, s):
        x, y = s
        return 0 <= x < self.size and 0 <= y < self.size

    def is_obstacle(self, s):
        return s in self.obstacles

    def move(self, state, action):
        x, y = state

        if action == "UP":
            nxt = (x, y-1)
        elif action == "DOWN":
            nxt = (x, y+1)
        elif action == "LEFT":
            nxt = (x-1, y)
        else:
            nxt = (x+1, y)

        if not self.in_bounds(nxt) or self.is_obstacle(nxt):
            return state

        return nxt

    def stochastic_action(self, action):
        r = random.random()
        if r < 0.8:
            return action
        elif r < 0.9:
            return self.left_perp(action)
        else:
            return self.right_perp(action)

    def left_perp(self, action):
        return {
            "UP": "LEFT",
            "DOWN": "RIGHT",
            "LEFT": "DOWN",
            "RIGHT": "UP"
        }[action]

    def right_perp(self, action):
        return {
            "UP": "RIGHT",
            "DOWN": "LEFT",
            "LEFT": "UP",
            "RIGHT": "DOWN"
        }[action]

    def step(self, state, action):
        action = self.stochastic_action(action)
        next_state = self.move(state, action)

        reward = -0.1
        done = False

        if next_state in self.traps:
            reward = -5

        if next_state in self.shortcut:
            next_state = self.shortcut[next_state]
            reward = self.shortcut_reward

        if next_state == self.goal:
            reward = 10
            done = True

        return next_state, reward, done

    def get_actions(self):
        return ACTIONS