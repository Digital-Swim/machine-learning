from importlib.resources import path
import random
import matplotlib.pyplot as plt
import numpy as np

from environments.world import World

# Grid with 
# - obstacles (black)
# - traps (red)
# - goal (green)
# - shortcut (blue)
#  Actions are stochastic: 80% intended, 10% left perp, 10% right perp
class GridWorld(World):
    
    arrow_map = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
    }
    
    def __init__(self, size = (10, 10), start=(0, 0), goal=(9, 9)):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = {(3,3), (3,4), (4,3)} 
        self.traps = {(5,5), (6,2)}
        self.shortcut = {(2,2): (7,7)}
        self.shortcut_reward = 2
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self):
        return self.start

    def in_bounds(self, s):
        x, y = s
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def is_obstacle(self, s):
        return s in self.obstacles

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
        
    def compute_value_grid(self, q_table):
        grid = np.zeros(self.size)

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                cell = (x, y)

                if cell in self.obstacles:
                    grid[x, y] = np.nan
                    continue

                q_values = [q_table.get((cell, a), 0.0) for a in self.actions]
                grid[x, y] = max(q_values)

        return grid

    def render(self, q_table=None, state=None, show_heatmap=False):
        self.init_figure()
        ax = self.ax
        ax.clear()

        if show_heatmap and q_table is not None:
            value_grid = self.compute_value_grid(q_table)

            im = ax.imshow(
                value_grid.T,
                extent=[0, self.size[0], self.size[1], 0],
                vmin=-5,
                vmax=10,
            )

            if not hasattr(self, "colorbar"):
                self.colorbar = self.fig.colorbar(im, ax=ax)

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                cell = (x, y)

                if cell in self.obstacles:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))
                elif cell in self.traps:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='red'))
                elif cell == self.goal:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='green'))
                elif cell in self.shortcut:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='blue'))
                else:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False))

                if q_table and cell not in self.obstacles:
                    q = {a: q_table.get((cell, a), 0.0) for a in self.actions}
                    best_action = max(q, key=q.get)
                    
                    ax.text(
                        x + 0.5,
                        y + 0.35,
                        self.arrow_map[best_action],
                        ha='center',
                        va='center',
                        fontsize=12,
                        fontweight='bold'
                    )

                    q_text = (
                        f"{q['UP']:.2f}\n"
                        f"{q['LEFT']:.2f} {q['RIGHT']:.2f}\n"
                        f"{q['DOWN']:.2f}"
                    )

                    ax.text(
                        x + 0.5,
                        y + 0.75,
                        q_text,
                        ha='center',
                        va='center',
                        fontsize=6
                    )

        if state:
            x, y = state
            ax.add_patch(plt.Circle((x+0.5, y+0.5), 0.3))

        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.set_xticks(range(self.size[0]))
        ax.set_yticks(range(self.size[1]))
        ax.grid()

        ax.invert_yaxis()
        
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
        return self.actions
                
    def is_terminal(self, state):
        return state == self.goal