import numpy as np

from navigation.src.utils import Actions


class RandomAgent(object):
  def get_action(self, timestep):
    return np.random.randint(4)


class FixedActionAgent(object):
  def __init__(self):
    self.t = -1
    self.actions = [Actions.UP] + [Actions.RIGHT]*7 + [Actions.DOWN]*4 + [Actions.LEFT]*4 \
                   + [Actions.DOWN]*3 + [Actions.RIGHT]*4 + [Actions.DOWN]
    self.tot_actions = len(self.actions)

  def get_action(self, timestep):
    self.t = (self.t + 1) % self.tot_actions
    return self.actions[self.t]


