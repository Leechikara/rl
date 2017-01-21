from torch_rl.policies.Policy import Policy
import torch_rl
import gym
import numpy as np
from torch.autograd import Variable

class DiscreteModelPolicy(Policy):
    '''A DiscreteModelPolicy is a policy that computes one score for each possible action given a particular torch model, and returns the action with the max value'''

    def __init__(self, action_space, torch_model):
        Policy.__init__(self, action_space)
        assert isinstance(action_space, gym.spaces.Discrete), "In DiscreteModelPolicy, the action space must be discrete"
        #assert isinstance(sensor_space, torch_rl.spaces.PytorchBox), "In DeeQPolicy, sensor_space must be a PytorchBox"

        self.action_space = action_space
        #self.sensor_space = sensor_space
        self.torch_model=torch_model

    # We assume that the observation is a 1xsingle observation
    def observe(self, observation):
        self.observation=observation.unsqueeze(0)
        pass

    # Sample an action
    def sample(self):
        scores=self.torch_model(Variable(self.observation))
        action=scores.max(1)
        action=action[1].data
        action=action[0][0]

        return action
