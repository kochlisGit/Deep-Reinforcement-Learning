import random as rand

class UniformReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.samples = []
        self.num_of_samples = 0

    # Adds a sample: (s,a,r,s',d) to agent's memory.
    # s:    State, converted into an observation.
    # a:    Action
    # r:    Reward
    # s':   Next State, converted into an observation.
    # d:    1 if Next State is Terminal, else 0.
    def add_sample(self, sample):
        self.samples.append(sample)
        if self.num_of_samples < self.memory_size:
            self.num_of_samples += 1
        else:
            self.samples.pop(0)

    # Returns a random mini-batch.
    def random_batch(self, batch_size):
        return rand.sample(self.samples, batch_size)
