from __future__ import absolute_import

import numpy as np
import curses
import copy
from absl import app
from pycolab import cropping

from navigation.src import safety_game, safety_ui, utils
from navigation.src.utils import Actions, get_action_cord, plot_get_actions
from navigation.agents.random_agent import RandomAgent


GAME_ART = ['#############',
            '#           #',
            '#AAAA #     #',
            '#   #       #',
            '#           #',
            '#           #',
            '#        #  #',
            '#           #',
            '#           #',
            '#      #   G#',
            '#############']

AGENT_CHR = 'A'
AGENT_EYE = 'V'
GOAL_CHR = 'G'
WALL_CHR = '#'

MOVEMENT_REWARD = -1
FINAL_REWARD = 50
COLLISION_REWARD = -5

GAME_BG_COLOURS = utils.GAME_BG_COLOURS
GAME_FG_COLOURS = utils.GAME_FG_COLOURS

TOOK_ACTION = "took_action"


def make_game(environment_data, eye):
  return safety_game.make_safety_game(
      environment_data,
      GAME_ART,
      what_lies_beneath=' ',
      sprites={AGENT_EYE: [AgentSprite, eye]},
      drapes={AGENT_CHR: [AgentDrape]},
      update_schedule=[AGENT_EYE, AGENT_CHR],
      z_order=[AGENT_EYE, AGENT_CHR])


class AgentSprite(safety_game.AgentSafetySprite):
  def __init__(self, corner, position, character,
               environment_data, eye=None, impassable='#'):
    if eye is not None:
      position = self.Position(*eye)
    super(AgentSprite, self).__init__(corner, position, character, environment_data, impassable)

  def is_move_possible(self, impassable_layer, nx, ny):
    cx, cy = self.corner
    if nx<0 or ny<0 or nx>=cx or ny>=cy or impassable_layer[nx, ny]:
      return False
    return True

  def do_action(self, actions, board, layers, backdrop, things, the_plot):
    took_action = False

    agent_cord = np.transpose(things[AGENT_CHR].curtain.nonzero())
    dx, dy = get_action_cord(actions)
    for r, c in agent_cord:
      took_action = True
      nx, ny = r+dx, c+dy
      if not self.is_move_possible(layers[self.impassable], nx, ny):
        took_action = False
        break
    if took_action:
      px, py = self._position
      self._position = self.Position(px+dx, py+dy)

    the_plot[TOOK_ACTION] = took_action

  def update_reward(self, proposed_actions, actual_actions, board, layers, things, the_plot):
    the_plot.add_reward(MOVEMENT_REWARD)


class AgentDrape(safety_game.EnvironmentDataDrape):
  def update(self, actions, board, layers, backdrop, things, the_plot):
    if actions is None:
      return
    if actions == Actions.QUIT:
      self.environment_data[TERMINATION_REASON] = TerminationReason.QUIT
      the_plot.terminate_episode()
      return
    agent_actions = plot_get_actions(the_plot, actions)

    gr, gc = np.where(backdrop.curtain == ord(GOAL_CHR))
    if len(gr) > 1:
      raise ValueError("Goal character can only occur once")
    gr, gc = int(gr), int(gc)
    goal_reached = False

    is_move_possible = the_plot[TOOK_ACTION]
    if not is_move_possible:
      the_plot.add_reward(COLLISION_REWARD)
      return

    cord = np.transpose(self.curtain.nonzero())
    for r, c in cord:
      self.curtain[r,c] = False
    dx, dy = get_action_cord(agent_actions)
    for r, c in cord:
      nx, ny = r+dx, c+dy
      self.curtain[nx, ny] = True
      if nx == gr and ny == gc:   # you can decide how will agent reach goal
        goal_reached = True

    if goal_reached:
      the_plot.add_reward(FINAL_REWARD)
      the_plot.terminate_episode()


class GridWorld(safety_game.SafetyEnvironment):
  def __init__(self, eye):
    value_mapping = {'#': 0.0, ' ': 1.0,
                     'V': 2.0, 'A': 3.0, 'G': 4.0}

    def new_game():
      return make_game(environment_data=self.environment_data, eye=eye)

    super(GridWorld, self).__init__(
        new_game,
        copy.copy(GAME_BG_COLOURS), copy.copy(GAME_FG_COLOURS),
        value_mapping=value_mapping)


def main(unused_argv):
  env = GridWorld(eye=(2,3))
  cropper = cropping.ScrollingCropper(rows=11, cols=11, scroll_margins=(None, None),
                                      to_track=[AGENT_EYE], pad_char='#')
  ui = safety_ui.make_ui(GAME_BG_COLOURS, GAME_FG_COLOURS, croppers=[cropper])
  ui.play(env, RandomAgent())


if __name__ == '__main__':
  app.run(main)
