# Original code: deepmind/ai-safety-gridworlds

import enum


class TerminationReason(enum.IntEnum):
  TERMINATED   = 0
  MAX_STEPS    = 1
  INTERRUPTED  = 2
  QUIT         = 3


class Actions(enum.IntEnum):
  UP    = 0
  DOWN  = 1
  LEFT  = 2
  RIGHT = 3
  NOOP  = 4
  QUIT  = 5


class SwitchActions(enum.IntEnum):
  COMPUTER = 6
  HUMAN    = 7


# Colours common in all environments.
GAME_BG_COLOURS = {' ': (858, 858, 858),  # Environment floor.
                   '#': (599, 599, 599),  # Environment walls.
                   'A': (0, 706, 999),    # Player character.
                   'G': (0, 823, 196)}    # Goal.
GAME_FG_COLOURS = {' ': (858, 858, 858),
                   '#': (599, 599, 599),
                   'A': (0, 0, 0),
                   'G': (0, 0, 0)}

# If not specified otherwise, these are the actions a game will use.
DEFAULT_ACTION_SET = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]

# Some constants to use with the environment_data dictionary to avoid
ENV_DATA = 'environment_data'
ACTUAL_ACTIONS = 'actual_actions'
HUMAN = 'human'
TERMINATION_REASON = 'termination_reason'
HIDDEN_REWARD = 'hidden_reward'

# Constants for the observations dictionary to the agent.
EXTRA_OBSERVATIONS = 'extra_observations'


def get_action_cord(action):
  ACTION_MAP = {
    Actions.UP: (-1, 0), #up
    Actions.DOWN: (1, 0), #down
    Actions.RIGHT: (0, 1), #right
    Actions.LEFT: (0, -1), #left
    Actions.NOOP: (0, 0),
  }
  return ACTION_MAP[action]


def plot_clear_actions(the_plot):
  if ACTUAL_ACTIONS in the_plot:
    del the_plot[ACTUAL_ACTIONS]


def plot_get_actions(the_plot, actions):
  return the_plot.get(ACTUAL_ACTIONS, actions)

