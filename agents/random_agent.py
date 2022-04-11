import numpy as np

from navigation.src.utils import Actions


class RandomAgent(object):
  def get_action(self, timestep):
    return np.random.choice(4)


