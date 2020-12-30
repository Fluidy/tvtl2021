import random


class ExpReplay:
    def __init__(self, mem_size, start_mem, batch_size):
        self.mem_size = mem_size
        self.start_mem = start_mem
        self.batch_size = batch_size
        self.mem = []

    def add_step(self, step):
        self.mem.append(step)
        if len(self.mem) > self.mem_size:
            self.mem.remove(self.mem[0])

    def sample(self):
        if len(self.mem) < self.start_mem:
            return []
        sampled_idx = random.sample(range(len(self.mem)), self.batch_size)
        samples = [self.mem[idx] for idx in sampled_idx]
        return samples
