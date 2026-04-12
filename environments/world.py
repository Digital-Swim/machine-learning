

from abc import ABC, abstractmethod


class World(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, state, action):
        pass
    
    @abstractmethod
    def show_policy(self, q_table):
        pass