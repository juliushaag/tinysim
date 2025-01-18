from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import mujoco
import mujoco.viewer as mjv
import numpy as np

from scipy.spatial.transform import Rotation as R

from tinysim.core.transform import Position, Rotation

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
  
  def render_point(self, pos : np.ndarray, color : np.ndarray, size : float):
    ...

class MjRenderer(SimulationRenderer):

  NAME = "mjviewer"

  def init_scene(self, sim, show_left_ui = False, show_right_ui = False):
    self.viewer = mjv.launch_passive(sim.model, sim.data, show_left_ui=show_left_ui, show_right_ui=show_right_ui)
    self.custom_object_count = 0
    self.viewer.user_scn.ngeom = 0

    self.debug_names = dict()

  def update_scene(self, sim):
    self.viewer.sync()

  def close(self, sim):
    self.viewer.close()

  def is_running(self):
    return self.viewer.is_running()
  
  def render_point(self, name : str,  pos : Position, color=np.array([1, 0, 0, 1]), size=np.array([0.2, 0.0, 0.0])):
    if name in self.debug_names:
      self.viewer.user_scn.geoms[self.debug_names[name]].pos = pos._data.numpy()
      return
    
    mujoco.mjv_initGeom(
        self.viewer.user_scn.geoms[self.custom_object_count],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=size,
        pos=pos._data.numpy(),
        mat=np.eye(3).flatten(),
        rgba=color
    )

    self.debug_names[name] = self.custom_object_count
    
    self.custom_object_count += 1
    self.viewer.user_scn.ngeom = self.custom_object_count


SimulationRenderer.register_backend(MjRenderer)