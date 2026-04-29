
from agents.agent import Agent


class RTDPAgent(Agent):
    def __init__(self, actions, alpha=0.1, gamma=0.9):
        super().__init__(actions)
        self.alpha = alpha
        self.gamma = gamma

    def learn(self, state, action, reward, next_state):
        best_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.get_q_value(state, action)
        new_q = self.get_q_value(state, action) + self.alpha * td_error
        self.q_table[(state, action)] = new_q

    def choose_action(self, state):
        # Choose the action with the highest Q-value for the current state
        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a in self.actions if self.get_q_value(state, a) == max_q]
        return best_actions[0]  # Return the first best action (deterministic choice)
