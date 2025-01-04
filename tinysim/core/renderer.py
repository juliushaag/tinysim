from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import mujoco.viewer as mjv
import numpy as np

from scipy.spatial.transform import Rotation as R


@dataclass
class RenderTransform:
  position: np.ndarray = field(default_factory=lambda: np.zeros(3))
  quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
  scale : np.ndarray = field(default_factory=lambda: np.ones(3))

class RenderPrimitiveType(str, Enum):
  CUBE = "CUBE"
  SPHERE = "SPHERE"
  CAPSULE = "CAPSULE"
  CYLINDER = "CYLINDER"
  PLANE = "PLANE"
  QUAD = "QUAD"
  MESH = "MESH"
  NONE = "NONE"

class SimulationRenderer():

  BACKENDS = dict()

  @classmethod
  def register_backend(cls, backend : "SimulationRenderer"):
    assert hasattr(backend, "NAME")

    cls.BACKENDS[backend.NAME] = backend

  @classmethod
  def create(cls, name : str, **kwargs) -> "SimulationRenderer":
    if not name in cls.BACKENDS: return SimulationRenderer()
    return cls.BACKENDS[name](**kwargs)


  def init_scene(self, sim):
    ...

  def update_scene(self, sim):
    ...

  def close(self, sim):
    ...

  def is_running(self):
    return True

class MjRenderer(SimulationRenderer):

  NAME = "mjviewer"

  def init_scene(self, sim, show_left_ui = False, show_right_ui = False):
    self.viewer = mjv.launch_passive(sim.model, sim.data, show_left_ui=show_left_ui, show_right_ui=show_right_ui)

  def update_scene(self, sim):
    self.viewer.sync()

  def close(self, sim):
    self.viewer.close()

  def is_running(self):
    return self.viewer.is_running()

SimulationRenderer.register_backend(MjRenderer)