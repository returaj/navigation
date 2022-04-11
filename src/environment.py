# Original code: deepmind/ai-safety-gridworlds
# Modified the original code and have kept necessary class/methods

from __future__ import absolute_import

import abc
import collections
import enum


class StepType(enum.IntEnum):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = 0
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = 1
  # Denotes the last `TimeStep` in a sequence.
  LAST = 2


class TimeStep(collections.namedtuple("TimeStep", ["state", "observation", "reward", "discount"])):
  __slots__ = ()

  def first(self):
    return self.state is StepType.FIRST

  def mid(self):
    return self.state is StepType.MID

  def last(self):
    return self.state is StepType.LAST


class Environment(object):
  """
  Please be nice to the objects of this class. Donot try to update internal fields of this class.
  Internal fields access have been provided under the condition that you will only use it for read.
  """
  def __init__(self, game_factory, all_actions,
               observation_distiller, max_iter=float('inf')):
    self.game_factory = game_factory
    self.all_actions = all_actions
    self.observation_distiller = observation_distiller
    self.max_iter = max_iter

    self.current_game = None
    self.state = None
    self.game_over = None

    self.last_observation = None
    self.last_reward = None
    self.last_discount = None

  def reset(self):
    self.current_game = self.game_factory()
    self.state = StepType.FIRST
    observation, reward, discount = self.current_game.its_showtime()
    self._update_for_game_step(observation, reward, discount)
    return TimeStep(self.state, self.last_observation, None, None)

  def step(self, action, continue_game=False):
    if self.game_over:
      if not continue_game:
        raise RuntimeError("Game has already ended. Please consider resetting the game.")
      # Reset the game 
      return self.reset()

    observation, reward, discount = self.current_game.play(action)
    self._update_for_game_step(observation, reward, discount)
    if self.game_over:
      self.state = StepType.LAST
    else:
      self.state = StepType.MID

    return TimeStep(self.state, self.last_observation, self.last_reward, self.last_discount)

  def _update_for_game_step(self, observation, reward, discount):
    self.last_observation = self.observation_distiller(observation)
    self.last_reward = reward
    self.last_discount = discount
    self.game_over = self.current_game.game_over
    if self.current_game.the_plot.frame >= self.max_iter:
      self.game_over = True

  def close(self):
    pass

  def __enter__(self):
    """Allows the environment to be used in a with-statement context."""
    return self

  def __exit__(self, unused_exception_type, unused_exc_value, unused_traceback):
    """Allows the environment to be used in a with-statement context."""
    self.close()


class Distiller(object):
  def __init__(self, repainter, array_converter):
    self.repainter = repainter
    self.array_converter = array_converter

  def __call__(self, observation):
    if self.repainter:
      observation = self.repainter(observation)
    return self.array_converter(observation)
