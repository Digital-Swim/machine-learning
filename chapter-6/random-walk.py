# Example 6.2: Random Walk of Machine Learning an Introduction to Reinforcement Learning  by Richard S. Sutton and Andrew G. Barto


import random

from matplotlib import pyplot as plt

states  = ['T1','A', 'B', 'C', 'D', 'E', 'T2']
actions = ['left', 'right']
start_state = 'C'
terminal_states = ['T1', 'T2']


# Example 6.3: Random Walk with TD(0) Update
# delta = alpha * (reward + gamma * V[next_state] - V[state])
# V[state] = V[state] + delta
# init state values to 0.5 except terminal states A and E ?? why can it be 0.5? because we know that A is 0 and E is 1, so the values in between should be around 0.5?
# can we init with 0.0? yes, but it will take more iterations to converge to the true values, because the initial estimates are further from the true values. The choice of 0.5 is a common heuristic that can speed up learning in this case, since it is closer to the true values of the non-terminal states.
class RandomWalkTD0:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.V = {s: 0.5 for s in states}


    def update(self, state, reward, next_state):
        self.V[state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
        
    def get_next_state(self, state, action):
        if state in terminal_states:
            return state
        if action == 'left':
            return states[states.index(state) - 1]
        elif action == 'right':
            return states[states.index(state) + 1]
    
    def get_reward(self, state):
        if state == 'T2':
            return 1
        else:
            return 0
    
    # probality of choosing an action is equal for both actions, so we can just choose a random action
    def choose_action(self, state):       
        return random.choice(actions)
    
    def run_episode(self):
        state = start_state
        while state not in terminal_states:
            action = self.choose_action(state=state)
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            self.update(state, reward, next_state)
            state = next_state        

class RandomWalkTD0Batch:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.V = {s: 0.5 for s in states}

    def update(self, episode):
        for state, reward, next_state in episode:
            self.V[state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
    
    def get_next_state(self, state, action):
        if state in terminal_states:
            return state
        if action == 'left':
            return states[states.index(state) - 1]
        elif action == 'right':
            return states[states.index(state) + 1]
    
    def get_reward(self, state):
        if state == 'T2':
            return 1
        else:
            return 0
    
    def choose_action(self, state):       
        return random.choice(actions)
    
    def run_episode(self):
        episode = []
        state = start_state
        while state not in terminal_states:
            action = self.choose_action(state=state)
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            episode.append((state, reward, next_state))
            state = next_state
        
        self.update(episode)    

# Monte Carlo update
# G = reward at the end of the episode
class RandomWalkMC:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.V = {s: 0.5 for s in states}

    def update(self, state, G):
        self.V[state] += self.alpha * (G - self.V[state])
    
    def get_next_state(self, state, action):
        if state in terminal_states:
            return state
        if action == 'left':
            return states[states.index(state) - 1]
        elif action == 'right':
            return states[states.index(state) + 1]
    
    def get_reward(self, state):
        if state == 'T2':
            return 1
        else:
            return 0
    
    def choose_action(self, state):       
        return random.choice(actions)
    
    def run_episode(self):
        episode = []
        state = start_state
        while state not in terminal_states:
            action = self.choose_action(state=state)
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(next_state)
            episode.append((state, reward))
            state = next_state
        
        G = 0
        for state, reward in reversed(episode):
            G = reward + self.gamma * G
            self.update(state, G)

class PlotEstimatedValues:
    def __init__(self, true_values):
        self.true_values = true_values
        self.estimated_values = {}
        self.estimated_values[-1] = true_values  # Add true values as a reference for episode -1

    def add_estimate(self, estimate, episode=0):
        #print(f"Episode {episode}: Estimated Values: {estimate}")
        self.estimated_values[episode] = estimate

    def plot(self):
        states = ['A', 'B', 'C', 'D', 'E']
        for episode, estimate in self.estimated_values.items():
            print(f"Episode {episode}: Estimated Values: {estimate}")
            plt.plot(states, [estimate[s] for s in states], label=f'Episode {episode}')  
       
        plt.xlabel('States')
        plt.ylabel('Value')
        plt.title('Estimated State Values (per Episode)')
        plt.legend()
        plt.show()
        
class PlotEmpiricalError:
    def __init__(self, true_values):
        self.true_values = true_values
        self.errors = {}  
        # structure: {algo: {alpha: [errors]}}

    def add_estimate(self, estimate, algo, alpha):
        if algo not in self.errors:
            self.errors[algo] = {}

        if alpha not in self.errors[algo]:
            self.errors[algo][alpha] = []

        #Empirical RMS error,averaged over states
        
        error = sum((estimate[s] - self.true_values[s]) ** 2 for s in self.true_values) / len(self.true_values)
        self.errors[algo][alpha].append(error)

    def plot(self):
        
        plt.figure(figsize=(14, 9))  # larger plot
        for algo, alpha_dict in self.errors.items():
            for alpha, error_list in alpha_dict.items():
                plt.plot(error_list, label=f'{algo}, alpha={alpha}')

        
        plt.xlabel('Episode')
        plt.ylabel('Mean Squared Error')
        plt.title('Empirical Error over Episodes')
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    
    alpha_values_td = [0.15, 0.1, 0.05]
    alpha_values_mc = [0.01, 0.02, 0.03]
    number_of_episodes = 10000
    true_values = {'A': 1/6, 'B': 2/6, 'C': 3/6, 'D': 4/6, 'E': 5/6} # true values for non-terminal states A, B, C, D, E
    plotter1 = PlotEmpiricalError(true_values=true_values)
    plotter2 = PlotEstimatedValues(true_values=true_values)
    
    for alpha in alpha_values_td:
        td0 = RandomWalkTD0(alpha=alpha)
        for episode in range(number_of_episodes):
            if episode in [0, 10, 100, 1000, 10000] and alpha == 0.1:  # plot estimates at specific episodes
                print(f"TD(0) with alpha={alpha} at episode {episode}: {td0.V}")
                plotter2.add_estimate(td0.V.copy(), episode=episode)
            td0.run_episode()
            
    for alpha in alpha_values_td:
        td0 = RandomWalkTD0(alpha=alpha)
        for episode in range(number_of_episodes):
            td0.run_episode()
            plotter1.add_estimate(td0.V, algo='TD(0)', alpha=alpha)
            
    #for alpha in alpha_values_mc:
    #    mc = RandomWalkMC(alpha=alpha)
    #    for episode in range(number_of_episodes):
    #        mc.run_episode()
    #        plotter.add_estimate(mc.V, algo='Monte Carlo', alpha=alpha)
                
    #plotter1.plot()
    plotter2.plot()
    
