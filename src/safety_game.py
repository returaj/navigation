# Original code: deepmind/ai-safety-gridworlds
# Modified the original code and have kept necessary class/methods

from __future__ import absolute_import

import abc

# Dependency imports
from navigation.src import observation_distiller
from navigation.src import environment as env
from navigation.src import utils
from navigation.src.utils import TerminationReason, Actions, plot_clear_actions, plot_get_actions

import enum
import numpy as np

from pycolab import ascii_art
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


class SafetyEnvironment(env.Environment):
  def __init__(self, game_factory, game_bg_colours,
               game_fg_colours, actions=None,
               value_mapping=None, environment_data=None,
               repainter=None, max_iter=100):

    if environment_data is None:
      environment_data = {}
    self.environment_data = environment_data

    self.episode_return = 0
    # Keys to clear from environment_data at start of each episode.
    self._keys_to_clear = [utils.TERMINATION_REASON, utils.ACTUAL_ACTIONS]

    if actions is None:
      actions = utils.DEFAULT_ACTION_SET

    if value_mapping is None:
      value_mapping = {chr(i) for i in range(256)}
    self.value_mapping = value_mapping

    array_converter = observation_distiller.ObservationToArrayWithRGB(
        value_mapping=value_mapping,
        colour_mapping=game_bg_colours)

    super(SafetyEnvironment, self).__init__(
      game_factory=game_factory, all_actions=actions,
      observation_distiller=env.Distiller(repainter, array_converter),
      max_iter=max_iter)

  def get_overall_performance(self):
    """
    Please consider overwriting this method.
    """
    return 0.5*self.episode_return + 0.5*self._get_hidden_reward()

  def _get_hidden_reward(self, default_reward=0):
    """Extract the hidden reward from the plot of the current episode."""
    return self.current_game.the_plot.get(utils.HIDDEN_REWARD, default_reward)

  def _clear_hidden_reward(self):
    """Delete hidden reward from the plot."""
    self.current_game.the_plot.pop(utils.HIDDEN_REWARD, None)

  def _get_agent_extra_observations(self):
    """Overwrite this method to give additional information to the agent."""
    return {}

  def reset(self):
    timestep = super(SafetyEnvironment, self).reset()
    return self._process_timestep(timestep)

  def step(self, actions, continue_game=False):
    timestep = super(SafetyEnvironment, self).step(actions, continue_game)
    return self._process_timestep(timestep)

  def _process_timestep(self, timestep):
    """
    Do timestep preprocessing before sending it to the agent.
    This method stores the cumulative return and makes sure that the
    `environment_data` is included in the observation.
    If you are overriding this method, make sure to call `super()` to include
    this code.
    Args:
      timestep: instance of environment.TimeStep
    Returns:
      Preprocessed timestep.
    """
    # Reset the cumulative episode reward.
    if timestep.first():
      self.episode_return = 0
      self._clear_hidden_reward()
      # Clear the keys in environment data from the previous episode.
      for key in self._keys_to_clear:
        self.environment_data.pop(key, None)
    # Add the timestep reward for internal wrapper calculations.
    if timestep.reward:
      self.episode_return += timestep.reward
    extra_observations = self._get_agent_extra_observations()
    if utils.ACTUAL_ACTIONS in self.environment_data:
      extra_observations[utils.ACTUAL_ACTIONS] = (
          self.environment_data[utils.ACTUAL_ACTIONS])
    if timestep.last():
      # Include the termination reason for the episode if missing.
      if utils.TERMINATION_REASON not in self.environment_data:
        self.environment_data[utils.TERMINATION_REASON] = TerminationReason.MAX_STEPS
      extra_observations[utils.TERMINATION_REASON] = (
          self.environment_data[utils.TERMINATION_REASON])
    timestep.observation[utils.EXTRA_OBSERVATIONS] = extra_observations
    return timestep


class SafetyBackdrop(plab_things.Backdrop):
  """The backdrop for the game.
  Clear some values in the_plot.
  """
  def update(self, actions, board, layers, things, the_plot):
    super(SafetyBackdrop, self).update(actions, board, layers, things, the_plot)
    plot_clear_actions(the_plot)


class AgentSafetySprite(plab_things.Sprite):
  def __init__(self, corner, position, character,
               environment_data, impassable='#'):
    super(AgentSafetySprite, self).__init__(corner, position, character)
    self.environment_data = environment_data
    self.impassable = impassable

  def update(self, actions, board, layers, backdrop, things, the_plot):
    if actions is None:
      return
    if actions == Actions.QUIT:
      self.environment_data[TERMINATION_REASON] = TerminationReason.QUIT
      the_plot.terminate_episode()
      return

    agent_actions = plot_get_actions(the_plot, actions)

    self.do_action(agent_actions, board, layers, backdrop, things, the_plot)
    self.update_reward(actions, agent_actions, board, layers, things, the_plot)

  def do_action(self, actions, board, layers, backdrop, things, the_plot):
    raise NotImplementedError

  def update_reward(self, proposed_actions, actual_actions, board, layers, things, the_plot):
    pass


class EnvironmentDataSprite(plab_things.Sprite):
  """
  A generic `Sprite` class for safety environments.
  All stationary Sprites in the safety environments should derive from this
  class.
  Its only purpose is to get hold of the environment_data dictionary variables.
  """

  def __init__(self, corner, position, character, environment_data):
    """Initialize environment data sprite.
    Args:
      corner: same as in pycolab sprite.
      position: same as in pycolab sprite.
      character: same as in pycolab sprite.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
    """
    super(EnvironmentDataSprite, self).__init__(corner, position, character)
    self.environment_data = environment_data

  def update(self, actions, board, layers, backdrop, things, the_plot):
    pass


class EnvironmentDataDrape(plab_things.Drape):
  """
  A generic `Drape` class for safety environments.
  All Drapes in the safety environments should derive from this class.
  Its only purpose is to get hold of the environment_data variables.
  """

  def __init__(self, curtain, character, environment_data):
    """
    Initialize environment data drape.
    Args:
      curtain: same as in pycolab drape.
      character: same as in pycolab drape.
      environment_data: dictionary of data that is passed to the pycolab
        environment and is used as a shared object that allows each wrapper to
        communicate with their environment.
    """
    super(EnvironmentDataDrape, self).__init__(curtain, character)
    self.environment_data = environment_data

  def update(self, actions, board, layers, backdrop, things, the_plot):
    pass


class PolicyWrapperDrape(EnvironmentDataDrape):
  action_key = utils.ACTUAL_ACTIONS

  def __init__(self, curtain, character, environment_data, agent_character):
    super(PolicyWrapperDrape, self).__init__(curtain, character, environment_data)
    self.agent_character = agent_character

  def update(self, actions, board, layers, backdrop, things, the_plot):
    agent_actions = plot_get_actions(the_plot, actions)
    if self.agent_character is not None:
      pos = things[self.agent_character].position
      # If the drape applies globally to all tiles instead of a specific tile,
      # redefine this function without the if statement on the following line.
      # (See example in 'whisky_gold.py.)
      if self.curtain[pos]:
        the_plot[cls.action_key] = self.get_actual_actions(agent_actions, things, the_plot)

  def get_actual_actions(self, actions, things, the_plot):
    """
    Takes the actions and returns new actions.
    A child `PolicyWrapperDrape` must implement this method.
    The PolicyWrapperDrapes are chained and can all change these actions.
    The actual actions returned by one drape are the actions input to the next
    one.
    See wkisky_gold.py (ai-safety-gridworld) for a usage example.
    Args:
      actions: either the actions output by the agent (if no drape have modified
        them), or the actions modified by a drape (policy wrapper).
      things: Sprites, Drapes, etc.
      the_plot: the Plot, as elsewhere.
    """
    raise NotImplementedError



def make_safety_game(environment_data, the_ascii_art,
                     what_lies_beneath, backdrop=SafetyBackdrop,
                     sprites=None, drapes=None,
                     update_schedule=None, z_order=None):
  """Create a pycolab game instance."""
  return ascii_art.ascii_art_to_game(
      the_ascii_art,
      what_lies_beneath,
      sprites=None if sprites is None
      else {k: ascii_art.Partial(args[0],
                                 environment_data,
                                 *args[1:])
            for k, args in sprites.items()},
      drapes=None if drapes is None
      else {k: ascii_art.Partial(args[0],
                                 environment_data,
                                 *args[1:])
            for k, args in drapes.items()},
      backdrop=backdrop,
      update_schedule=update_schedule,
      z_order=z_order)
