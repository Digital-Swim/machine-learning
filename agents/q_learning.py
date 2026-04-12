
import random
from agents.agent import Agent

class QLearningAgent(Agent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)

    # Q-learning update rule
    # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(next_state)) - Q(s, a))
    # Given a state, action, reward, and next state, update the Q-table using the Q-learning update rule
    def learn(self, state, action, reward, next_state):
        # Get the maximum Q-value for the next state
        best_next_q = max(self.get_q_value(next_state, a) for a in self.actions)
        # Get the delta for the Q-value update, which is the difference between the target 
        # (reward + gamma * best_next_q) and the current Q-value for the state-action pair
        delta = reward + self.gamma * best_next_q - self.get_q_value(state, action)
        # Update the Q-value for the state-action pair using the learning rate (alpha) and the delta
        new_q = self.get_q_value(state, action) + self.alpha * delta
        # Update the Q-table with the new Q-value
        self.q_table[(state, action)] = new_q
    
    # Choose an action based on the epsilon-greedy policy5 
    def choose_action(self, state, epsilon=0.1):
        # Explore: choose a random action
        if random.random() < epsilon:
            return random.choice(self.actions)
        # Exploit: choose the action with the highest Q-value for the current state
        else:
            # Get Q-values for all actions in the current state
            q_values = [self.get_q_value(state, a) for a in self.actions]
            # Get the maximum Q-value
            max_q = max(q_values)
            # Get all actions that have the maximum Q-value
            best_actions = [a for a in self.actions if self.get_q_value(state, a) == max_q]
            # If there are multiple best actions, choose one at random
            return random.choice(best_actions)