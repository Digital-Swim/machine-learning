
# Episodes 
# A, 0, B, 0 
# B, 1
# B, 1
# B, 1
# B, 1
# B, 1
# B, 1
# B, 0

# This code implements a batch update for the Random Walk problem using TD(0) learning. The agent updates its value estimates for each state based on the episodes it has experienced. The value of
# True values for A and B should converge to 3/4 
# As B is 1 in 6 out of 8 episodes, and 0 in 2 out of 8 episodes, the expected value for B is (6/8)*1 + (2/8)*0 = 3/4. The value for A should also converge to 3/4, as it leads to B which has a value of 3/4.
# Running this code will show the estimated values for states A and B after a specified number of episodes, which should converge towards the true values as more episodes are processed.
# This shows how batch updates can be used to learn value estimates from a set of episodes, and how the estimates converge to the true values over time.
# Here we are running same data sets repeatedly, so the values will converge to the true values as we run more episodes. The more episodes we run, the closer the estimates will get to the true values of 3/4 for both A and B.

episodes = [
    [('A', 0, 'B'), ('B', 0, 'T')],
    [('B', 1, 'T')],
    [('B', 1, 'T')],
    [('B', 1, 'T')],
    [('B', 1, 'T')],
    [('B', 1, 'T')],
    [('B', 1, 'T')],
    [('B', 0, 'T')]
]   

states = ['A', 'B', 'T']

# Batch update
class RandomWalkTD0Batch:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.V = {s: 0.5 for s in states}
        self.V['T'] = 0.0

    def update(self, episode):
        for state, reward, next_state in episode:
            self.V[state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])
    
    def run_episode(self):
        for episode in episodes:
            self.update(episode)


class MonteCarloBatch:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.V = {s: 0.5 for s in states}
        self.V['T'] = 0.0

    
    def update(self, episode):
        G = 0
        for state, reward, next_state in reversed(episode):
            G = reward + self.gamma * G
            self.V[state] += self.alpha * (G - self.V[state])
    
    def run_episode(self):
        for episode in episodes:
            self.update(episode)

if __name__ == "__main__":
    #agent = RandomWalkTD0Batch()
    agent = MonteCarloBatch()
    number_of_episodes = int(input("Enter the number of episodes to run: "))
    for _ in range(number_of_episodes):
        agent.run_episode()
    print(agent.V)