import random

class TransitionMemory:
    '''
    Implements a memory of transitions (s,a,s',r)
    When full, new transitions are added randomly in the memory
    '''

    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]

    def push(self,s,a,sprime,r=None):
        if (len(self.memory)==self.capacity):
            idx=random.randint(0,self.capacity-1)
            self.memory[idx]=(s,a,sprime,r)
        else:
            self.memory.append((s,a,sprime,r))

    def get(self):
        idx = random.randint(0, len(self.memory)- 1)
        return self.memory[idx]

    def size(self):
        return len(self.memory)