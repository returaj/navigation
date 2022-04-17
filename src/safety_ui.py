# Original code: deepmind/ai-safety-gridworlds
# Modified the original code and have kept necessary class/methods

"""Frontends for humans who want to play pycolab games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import datetime
import sys

# Dependency imports
from navigation.src import safety_game, utils
from navigation.src.utils import Actions, SwitchActions

from pycolab import human_ui
from pycolab.protocols import logging as plab_logging


class SafetyCursesUi(human_ui.CursesUi):
  def __init__(self, *args, **kwargs):
    super(SafetyCursesUi, self).__init__(*args, **kwargs)
    self._env = None
    self._computer = None

  def play(self, env, computer=None):
    if not isinstance(env, safety_game.SafetyEnvironment):
      raise ValueError('`env` must be an instance of `SafetyEnvironment`.')
    if self._game is not None:
      raise RuntimeError('CursesUi is not at all thread safe')
    self._env = env
    self._computer = computer

    self._timestep = self._env.reset()
    self._game = self._env.current_game
    self._start_time = datetime.datetime.now()

    # Inform the croppers which game we're playing.
    for cropper in self._croppers:
      cropper.set_engine(self._game)

    # start with human action
    self._env.environment_data[utils.HUMAN] = True

    # After turning on curses, set it up and play the game.
    curses.wrapper(self._init_curses_and_play)

    # The game has concluded. Print the final statistics.
    score = self._env.episode_task_return
    duration = datetime.datetime.now() - self._start_time
    termination_reason = self._env.environment_data[utils.TERMINATION_REASON]
    safety_performance = self._env.get_overall_performance()
    print('Game over! Final score is {}, earned over {}.'.format(
        score, _format_timedelta(duration)))
    print('Termination reason: {!s}'.format(termination_reason))

    # Clean up in preparation for the next game.
    self._timestep = None
    self._game = None
    self._start_time = None

  def _init_curses_and_play(self, screen):
    for key, action in self._keycodes_to_actions.items():
      if key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
        raise ValueError(
            'the keys_to_actions argument to the CursesUi constructor binds '
            'action {} to the {} key, which is reserved for CursesUi. Please '
            'choose a different key for this action.'.format(
                repr(action), repr(curses.keyname(key))))

    self._init_colour()
    curses.curs_set(0)  # We don't need to see the cursor.
    if self._delay is None:
      screen.timeout(-1)  # Blocking reads
    else:
      screen.timeout(self._delay)  # Nonblocking (if 0) or timing-out reads

    # Create the curses window for the log display
    rows, cols = screen.getmaxyx()
    console = curses.newwin(rows // 2, cols, rows - (rows // 2), 0)

    # By default, the log display window is hidden
    paint_console = False

    def crop_and_repaint(observation):
      observations = [cropper.crop(observation) for cropper in self._croppers]
      if self._repainter:
        if len(observations) == 1:
          return [self._repainter(observations[0])]
        else:
          return [copy.deepcopy(self._repainter(obs)) for obs in observations]
      else:
        return observations

    observation = self._game._board  # pylint: disable=protected-access
    observations = crop_and_repaint(observation)
    self._display(screen, observations, self._env.episode_task_return,
                  elapsed=datetime.timedelta())

    # Oh boy, play the game!
    while not self._env.game_over:  # pylint: disable=protected-access
      action = None
      keycode = screen.getch()
      if keycode == curses.KEY_PPAGE:    # Page Up? Show the game console.
        paint_console = True
      elif keycode == curses.KEY_NPAGE:  # Page Down? Hide the game console.
        paint_console = False

      if keycode in self._keycodes_to_actions:
        key_action = self._keycodes_to_actions[keycode]
        if key_action == SwitchActions.HUMAN:
          self._env.environment_data[utils.HUMAN] = True
        elif key_action == SwitchActions.COMPUTER:
          self._env.environment_data[utils.HUMAN] = False
        else:
          action = key_action

      # overwrite action if computer is suppose to act 
      if not self._env.environment_data[utils.HUMAN]:
        action = self._computer.get_action(self._timestep)

      if action is not None:
        self._timestep = self._env.step(action)
        observation = self._game._board  # pylint: disable=protected-access
        observations = crop_and_repaint(observation)

      # Update the game display, regardless of whether we've called the game's
      # play() method.
      elapsed = datetime.datetime.now() - self._start_time
      self._display(screen, observations, self._env.episode_task_return, elapsed)

      # Update game console message buffer with new messages from the game.
      self._update_game_console(
          plab_logging.consume(self._game.the_plot), console, paint_console)

      # Show the screen to the user.
      curses.doupdate()


def make_ui(game_bg_colours, game_fg_colours, delay=100, croppers=None):
  return SafetyCursesUi(
      keys_to_actions={curses.KEY_UP: Actions.UP,
                       curses.KEY_DOWN: Actions.DOWN,
                       curses.KEY_LEFT: Actions.LEFT,
                       curses.KEY_RIGHT: Actions.RIGHT,
                       'q': Actions.QUIT,
                       'h': SwitchActions.HUMAN,
                       'c': SwitchActions.COMPUTER},
      delay=delay,
      repainter=None,
      colour_fg=game_fg_colours,
      colour_bg=game_bg_colours,
      croppers=croppers)


def _format_timedelta(timedelta):
  """Convert timedelta to string, lopping off microseconds."""
  # This approach probably looks awful to all you time nerds, but it will work
  # in all the locales we use in-house.
  return str(timedelta).split('.')[0]