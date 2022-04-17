from __future__ import absolute_import

from absl import app
from pycolab import cropping

from navigation.environments import sub_prob_1
from sub_prob_1 import GridWorld, WALL_CHR, AGENT_CHR, AGENT_EYE, GAME_BG_COLOURS, GAME_FG_COLOURS
from navigation.src import safety_ui
from navigation.agents.example_agent import RandomAgent


def main(unused_argv):
  value_mapping = {'#': 0.0, ' ': 1.0, 'V': 2.0, 'A': 3.0, 'G': 4.0}
  cropper = cropping.ScrollingCropper(rows=11, cols=11, scroll_margins=(None, None),
                                      to_track=[AGENT_EYE], pad_char='#')
  env = GridWorld(wall_chr=WALL_CHR, agent_chr=AGENT_CHR, eye_chr=AGENT_EYE,
                  eye_pos=(2,4), value_mapping=value_mapping, cropper=cropper)
  ui = safety_ui.make_ui(GAME_BG_COLOURS, GAME_FG_COLOURS, croppers=[cropper])
  ui.play(env, RandomAgent())


if __name__ == '__main__':
  app.run(main)
