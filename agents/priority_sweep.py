import heapq

from agents.q_learning import QLearningAgent

class PrioritySweepAgent(QLearningAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, theta=0.01):
        super().__init__(actions, alpha, gamma, epsilon)
        self.priority_queue = []  # Queue for prioritized sweeping
        self.theta = theta  # Threshold for prioritized sweeping
        self.model = {}  # Model for planning, similar to Dyna-Q
        self.planning_steps = 10  # Number of planning steps to perform after each real experience
        self.predecessors = {}  # To keep track of predecessors for each state-action pair

    def learn(self, state, action, reward, next_state):
        # Perform the standard Q-learning update
        super().learn(state, action, reward, next_state)

        self.model[(state, action)] = (reward, next_state)  # Update the model with the new experience
        
        if next_state not in self.predecessors:
            self.predecessors[next_state] = set()  # Initialize predecessors for this state if not already present
        
        self.predecessors[next_state].add((state, action))  # Add the current state-action pair as a predecessor of the next state
        
    
        self._push_priority(state, action, reward, next_state)
            

        # Perform prioritized sweeping updates
        self.prioritized_sweep()

    def prioritized_sweep(self):
        for _ in range(self.planning_steps):
            if not self.priority_queue:
                break
            
            # Get the state-action pair with the highest priority (largest TD error)
            _, state, action = heapq.heappop(self.priority_queue)
            
            # Get the reward and next state from the model for the chosen state-action pair
            reward, next_state = self.model[(state, action)]
            
            # Perform a Q-learning update for this state-action pair using the model's reward and next state
            super().learn(state, action, reward, next_state)
            
            # If the TD error for this update is above the threshold, add its predecessors to the priority queue
            if (state in self.predecessors):
                for pred_state, pred_action in self.predecessors[state]:
                    pred_reward, pred_next_state = self.model[(pred_state, pred_action)]
                    self._push_priority(pred_state, pred_action, pred_reward, pred_next_state)
    
    def _push_priority(self, state, action, reward, next_state):
        # Calculate the TD error for the state-action pair
        td_error = abs(reward + self.gamma * self.get_max_q(next_state) - self.get_q_value(state, action))
        
        # If the TD error exceeds the threshold, add it to the priority queue
        if td_error > self.theta:
            # negate the TD error to create a max-heap based on the absolute value of the TD error
            heapq.heappush(self.priority_queue, (-td_error, state, action))      
        
        

    def get_max_q(self, state):
        return max([self.get_q_value(state, a) for a in self.actions])