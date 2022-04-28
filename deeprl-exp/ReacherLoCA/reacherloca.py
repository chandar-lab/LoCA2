# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""ReacherLoCA domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('reacherloca.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns ReacherLoCA with sparse reward with 5e-2 tol."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = ReacherLoCA(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns ReacherLoCA with sparse reward with 1e-2 tol."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = ReacherLoCA(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the ReacherLoCA domain."""

  def finger_to_targets(self):
    """Returns the vectors from targets to finger in global coordinates."""
    finger_to_target_1 = (self.named.data.geom_xpos['target_1', :2] - self.named.data.geom_xpos['finger', :2])
    finger_to_target_2 = (self.named.data.geom_xpos['target_2', :2] - self.named.data.geom_xpos['finger', :2])
    return (finger_to_target_1, finger_to_target_2)

  def finger_to_targets_dist(self):
    """Returns the signed distance between the finger and targets surface."""
    finger_to_target_1, finger_to_target_2 = self.finger_to_targets()
    return (np.linalg.norm(finger_to_target_1), np.linalg.norm(finger_to_target_2))


class ReacherLoCA(base.Task):
  """A ReacherLoCA `Task` to reach the targets."""

  def __init__(self, target_size, random=None):
    """Initialize an instance of `ReacherLoCA`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._target_size = target_size
    self._target_1_r = 4 
    self._target_2_r = 2
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target_1', 0] = self._target_size
    physics.named.model.geom_size['target_2', 0] = self._target_size
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # Targets position
    angle_1 = 0.25 * np.pi
    angle_2 = 1.25 * np.pi
    radius = 0.15
    physics.named.model.geom_pos['target_1', 'x'] = radius * np.sin(angle_1)
    physics.named.model.geom_pos['target_1', 'y'] = radius * np.cos(angle_1)
    physics.named.model.geom_pos['target_2', 'x'] = radius * np.sin(angle_2)
    physics.named.model.geom_pos['target_2', 'y'] = radius * np.cos(angle_2)

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state and the target position."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_targets'] = np.array(physics.finger_to_targets())
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    radii = physics.named.model.geom_size[['target_1', 'finger'], 0].sum()
    finger_to_target_1_dist, finger_to_target_2_dist = physics.finger_to_targets_dist()
    return self._target_1_r * rewards.tolerance(finger_to_target_1_dist ,(0, radii)) + \
       self._target_2_r * rewards.tolerance(finger_to_target_2_dist ,(0, radii))

  def switch_loca_task(self):
    if self._target_1_r == 4:
      self._target_1_r = 1
    else:
      self._target_1_r = 4
