
from abc import ABC, abstractmethod
import json
import csv

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
    
    def save_q_table(self, filename="q_table.json"):
        data = [
            {
                "state": state,
                "action": action,
                "value": value
            }
            for (state, action), value in self.q_table.items()
        ]

        with open(filename, "w") as f:
            json.dump(data, f)      
    
    import csv

    def export_q_table(self, filename="q_table.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # header
            writer.writerow([
                "velocity",
                "distance",
                "width",
                "jumpVelocity",
                "action",
                "q_value"
            ])

            for (state, action), value in self.q_table.items():
                v, d, w, jumping = state

                writer.writerow([v, d, w, jumping, action, value])