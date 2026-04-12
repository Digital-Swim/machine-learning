
import random
from agents.q_learning import QLearningAgent

# Dyna-Q agent that combines Q-learning with a model-based approach for planning
# The Dyna-Q agent learns from real experience using the Q-learning update rule 
# and also updates its Q-values based on simulated experience generated from a learned model of the environment
# Learn a step from real experience and then perform planning steps using the learned model to update Q-values 

class DyncaQAgent(QLearningAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)
        # Memory for storing the model of the environment, 
        # which maps state-action pairs to (reward, next_state) tuples based on real experience
        self.model = {}  # Model for planning

    def learn(self, state, action, reward, next_state):
        
        # First, perform the Q-learning update based on the real experience
        super().learn(state, action, reward, next_state)
        
        # Then, update the model with the new experience, which will be used for planning
        self.model[(state, action)] = (reward, next_state)  # Update model
        
        # Now permorm planning steps using the learned model to update Q-values based on simulated experience
        self.plan()  # Perform planning steps after learning from real experience

    def plan(self, n=10):
        for _ in range(n):
            if not self.model:
                break
            # Choose a random state-action pair from the model to simulate experience
            state_action = random.choice(list(self.model.keys()))
            
            # Get the reward and next state from the model for the chosen state-action pair
            reward, next_state = self.model[state_action]
            
            super().learn(state_action[0], state_action[1], reward, next_state)