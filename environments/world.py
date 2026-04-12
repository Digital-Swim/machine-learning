from abc import ABC, abstractmethod

class World(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, state, action):
        pass

    @abstractmethod
    def render(self, q_table=None, state=None, show_heatmap=False):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def is_terminal(self, state):
        pass

    def compute_state_value(self, state, q_table):
        return max([q_table.get((state, a), 0.0) for a in self.get_actions()])

    def get_best_action(self, state, q_table):
        q = {a: q_table.get((state, a), 0.0) for a in self.get_actions()}
        return max(q, key=q.get)

    def show_policy(self, q_table):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        self.render(q_table=q_table)
        plt.show()

    def animate_policy(self, q_table, max_steps=100, delay=0.3, deterministic=False):
        import matplotlib.pyplot as plt

        state = self.reset()
        plt.figure(figsize=(6,6))
        plt.ion()

        for _ in range(max_steps):
            self.render(q_table=q_table, state=state)
            plt.pause(delay)

            action = self.get_best_action(state, q_table)

            if deterministic:
                next_state = self.move(state, action)
                done = self.is_terminal(next_state)
            else:
                next_state, _, done = self.step(state, action)

            state = next_state

            if done:
                self.render(q_table=q_table, state=state)
                break

        plt.ioff()
        plt.show()

    def animate_training(self, agent, episodes=100, delay=0.1):
        import matplotlib.pyplot as plt

        plt.ion()

        for _ in range(episodes):
            state = self.reset()
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = self.step(state, action)

                agent.learn(state, action, reward, next_state)
                state = next_state

                self.render(q_table=agent.q_table, state=state, show_heatmap=True)
                plt.pause(delay)

        plt.ioff()
        plt.show()
              
    def train(self, agent, episodes=1000, visualize=False, delay=0.1, show_heatmap=False):
        import matplotlib.pyplot as plt

        if visualize:
            plt.ion()

        for _ in range(episodes):
            state = self.reset()
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = self.step(state, action)

                agent.learn(state, action, reward, next_state)
                state = next_state

                if visualize:
                    self.render(q_table=agent.q_table, state=state, show_heatmap=show_heatmap)
                    plt.pause(delay)

        if visualize:
            plt.ioff()
            plt.show()