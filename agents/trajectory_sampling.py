
from agents.q_learning import QLearningAgent


class TrajectorySamplingAgent(QLearningAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)
        self.trajectory_length = 5  # Length of the trajectory to sample for updates
        self.model = {}  # Model for planning, similar to Dyna-Q
        
        
    def learn(self, state, action, reward, next_state):
        # Perform the standard Q-learning update
        super().learn(state, action, reward, next_state)
        
        self.model[(state, action)] = (reward, next_state)  # Update the model with the new experience, which will be used for trajectory sampling
        
        
        self.sample_trajectory(state, action)  # Sample a trajectory from the model and update Q-values based on the sampled trajectory
        
    def sample_trajectory(self, start_state, start_action):
        trajectory = []
        current_state = start_state
        current_action = start_action
        
        # Sample a trajectory of a fixed length from the model
        for _ in range(self.trajectory_length):
            if (current_state, current_action) not in self.model:
                break  # Stop if we reach a state-action pair that is not in the model
            
            reward, next_state = self.model[(current_state, current_action)]
            trajectory.append((current_state, current_action, reward, next_state))
            
            # Choose the next action based on the epsilon-greedy policy
            current_state = next_state
            current_action = self.choose_action(current_state)
        
        # Update Q-values based on the sampled trajectory using the Q-learning update rule
        for state, action, reward, next_state in reversed(trajectory):
            super().learn(state, action, reward, next_state)
    