from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')  # no UI backend
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque, namedtuple
from pycolab import cropping 

from navigation.environments.sub_prob_1 import GridWorld, WALL_CHR, SPACE_CHR, AGENT_CHR, AGENT_EYE, GOAL_CHR


SEED = 0
np.random.seed(SEED)
random.seed(SEED)


class ReplayBuffer(object):
  def __init__(self, buffer_size):
    self.memory = deque(maxlen=buffer_size)
    self.experience = namedtuple("Experience", field_names=["state", "action", "task_reward", "safe_reward", "next_state", "done"])

  def __len__(self):
    return len(self.memory)

  def clear(self):
    self.memory.clear()

  def add(self, state, action, task_reward, safe_reward, next_state, done):
    e = self.experience(state, action, task_reward, safe_reward, next_state, done)
    self.memory.append(e)

  def sample(self, batch_size):
    return random.sample(self.memory, k=batch_size)


class ValueDecomposition(object):
  def __init__(self, env, buffer_size, beta=0.4, temp=5.0, temp_decay=0.99, alpha=0.01, gamma=1.0):
    self.env = env
    self.num_states = env.rows * env.cols
    self.actions = env.all_actions
    self.num_actions = len(self.actions)
    self.buffer = ReplayBuffer(buffer_size)
    self.gamma = gamma
    self.beta = beta
    self.temp = temp
    self.temp_decay = temp_decay
    self.alpha = alpha

  def encode(self, timestep):
    pos = timestep.agent_eye
    r, c = self.env.rows, self.env.cols
    return pos.row*c + pos.col

  def softmax(self, ts, qvalue, temp):
    state = self.encode(ts)
    qval = qvalue[state] / temp
    tmp_prob = np.exp(qval - np.max(qval))
    prob = tmp_prob / np.sum(tmp_prob)
    return prob

  def get_action(self, ts, tq, sq, temp):
    prob_t = self.softmax(ts, tq, temp)
    prob_s = self.softmax(ts, sq, temp)
    prob = self.beta * prob_t + (1-self.beta) * prob_s
    return np.random.choice(self.actions, p=prob)

  def add_to_buffer(self, timestep, action, next_timestep):
    state       = self.encode(timestep)
    task_reward = next_timestep.task_reward
    safe_reward = next_timestep.safe_reward
    next_state  = self.encode(next_timestep)
    done        = next_timestep.last()
    self.buffer.add(state, action, task_reward, safe_reward, next_state, done)

  def update_q(self, tq, sq, batch_size):
    if len(self.buffer) < batch_size:
      return
    experiences = self.buffer.sample(batch_size)
    for exp in experiences:
      task_target = exp.task_reward + self.gamma*np.max(tq[exp.next_state])
      tq[exp.state, exp.action] += self.alpha * (task_target - tq[exp.state, exp.action])
      safe_target = exp.safe_reward + self.gamma*np.max(sq[exp.next_state])
      sq[exp.state, exp.action] += self.alpha * (safe_target - sq[exp.state, exp.action])

  def single_run(self, episodes=100, batch_size=32):
    env = self.env
    temp = self.temp
    self.buffer.clear()
    tq = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
    sq = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
    task_episodes, safe_episodes = np.zeros(episodes), np.zeros(episodes)
    for episode in range(episodes):
      timestep = env.reset()
      trwd, srwd = 0, 0
      temp = max(0.1, temp*self.temp_decay)
      while not timestep.last():
        action = self.get_action(timestep, tq, sq, temp)
        next_timestep = env.step(action)
        self.add_to_buffer(timestep, action, next_timestep)
        self.update_q(tq, sq, batch_size)
        trwd += next_timestep.task_reward
        srwd += next_timestep.safe_reward
        timestep = next_timestep
      task_episodes[episode] = trwd
      safe_episodes[episode] = srwd
      # print(env.episode_task_return)
    return tq, sq, task_episodes, safe_episodes

  def avg_run(self, runs=10, episodes=100):
    avg_tq = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
    avg_sq = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
    avg_task, avg_safe = np.zeros(episodes), np.zeros(episodes)
    for run in range(runs):
      tq, sq, task, safe = self.single_run(episodes)
      avg_tq += (tq - avg_tq) / (run + 1)
      avg_sq += (sq - avg_sq) / (run + 1)
      avg_task += (task - avg_task) / (run + 1)
      avg_safe += (safe - avg_safe) / (run + 1)
    return avg_tq, avg_sq, avg_task, avg_safe

  def plot(self, value, name):
    episodes = np.arange(len(value))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(episodes, value)
    ax.set_title(f"episode-{name}-plot")
    ax.set_xlabel("episode")
    ax.set_ylabel(name)
    plt.savefig(f"{name}.png")

  def save_q_values(self, tq, sq):
    np.save('value_decomposition_task_q', tq)
    np.save('value_decomposition_safe_q', sq)


if __name__ == '__main__':
  value_mapping = {SPACE_CHR: 0.0, WALL_CHR: 1.0, AGENT_EYE: 2.0, AGENT_CHR: 3.0, GOAL_CHR: 4.0}
  env = GridWorld(WALL_CHR, AGENT_CHR, AGENT_EYE, (2,4), value_mapping, max_iter=700)
  runner = ValueDecomposition(env, int(1e3), beta=0.4, temp=5.0, temp_decay=0.99, alpha=0.01) # beta=0.4(safe), beta=1.0(greedy)
  # save avg reward plot
  avg_tq, avg_sq, avg_task, avg_safe = runner.avg_run(episodes=1_000, runs=1)
  runner.save_q_values(avg_tq, avg_sq)
  runner.plot(avg_task, "task_reward")
  runner.plot(avg_safe, "safe_reward")
