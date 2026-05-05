import random
from time import sleep, time

from agents.agent import Agent
from agents.dyna_q import DyncaQAgent
from agents.priority_sweep import PrioritySweepAgent
from agents.q_learning import QLearningAgent
from agents.trajectory_sampling import TrajectorySamplingAgent
from environments.dino import DinoWorld
from environments.grid_world import GridWorld
from environments.world import World


def train_agent(agent: Agent, env:World, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(state, action)
            agent.learn(state, action, reward, next_state)
            state = next_state 
            env.show_policy(agent.q_table)
            
            
def env_test():
    env = GridWorld()
    state = env.reset()
    for _ in range(20):
        action = random.choice(env.get_actions())
        next_state, reward, done = env.step(state, action)
        print(state, action, "->", next_state, reward)
        state = next_state
        if done:
            break        
            

def DinoTestQLearning():
    env = DinoWorld()
    env.reset()            
    agent = QLearningAgent(actions=env.get_actions())
    env.train(agent, episodes=100, visualize=False, delay=0.1, show_heatmap=False)
    

def QLearningTest():
    env = GridWorld(start=(0, 0), goal=(9, 9))
    agent = QLearningAgent(actions=env.get_actions())
    env.train(agent, episodes=100, visualize=True, delay=0.1, show_heatmap=True)
    

def DynaQTest():
    env = GridWorld(start=(9, 0), goal=(9, 9))
    agent = DyncaQAgent(actions=env.get_actions())
    env.train(agent, episodes=20, visualize=True, delay=0.1, show_heatmap=True)

def PrioritySweepTest():
    env = GridWorld(start=(0, 0), goal=(9, 9))
    agent = PrioritySweepAgent(actions=env.get_actions())
    env.train(agent, episodes=100, visualize=True, delay=0.05, show_heatmap=True)

def TrajectoryTest():
    env = GridWorld(start=(0, 0), goal=(9, 9))
    agent = TrajectorySamplingAgent(actions=env.get_actions()) 
    env.train(agent, episodes=100, visualize=True, delay=0.05, show_heatmap=True)

if __name__ == "__main__":
    # env_test()
    # QLearningTest()
    # DynaQTest()
    #PrioritySweepTest()
    DinoTestQLearning()