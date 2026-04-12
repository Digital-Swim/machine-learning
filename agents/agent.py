
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod     
    def learn(self, state, action, reward, next_state):
        pass
    