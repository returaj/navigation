import numpy as np

from navigation.src.utils import Actions, DEFAULT_ACTION_SET


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


VALUE_DECOMPOSITION_TASK_PATH = '/mnt/d/books/iitm/hlrl/codes/navigation/output/game_art_2/safe_0.4/value_decomposition_task_q.npy'
VALUE_DECOMPOSITION_SAFE_PATH = '/mnt/d/books/iitm/hlrl/codes/navigation/output/game_art_2/safe_0.4/value_decomposition_safe_q.npy'

class ValueDecompositionAgent(object):
  def __init__(self, env, task_filepath=None, safe_filepath=None, beta=0.9):
    if task_filepath is None:
      task_filepath = VALUE_DECOMPOSITION_TASK_PATH
    if safe_filepath is None:
      safe_filepath = VALUE_DECOMPOSITION_SAFE_PATH
    self.tq = np.load(task_filepath)
    self.sq = np.load(safe_filepath)
    self.env = env
    self.actions = DEFAULT_ACTION_SET
    self.beta = beta

  def encode(self, pos):
    r, c = self.env.rows, self.env.cols
    return pos.row*c + pos.col

  def softmax(self, qvalue):
    qvalue /= 0.2
    tmp_prob = np.exp(qvalue - np.max(qvalue)) 
    prob = tmp_prob / np.sum(tmp_prob)
    return prob

  def get_action(self, timestep):
    state = self.encode(timestep.agent_eye)
    prob = self.beta * self.softmax(self.tq[state]) + (1-self.beta) * self.softmax(self.sq[state])
    action = np.random.choice(self.actions, p=prob)
    return action
