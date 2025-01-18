import time
import math
import mujoco as mj
import numpy as np
import torch

from tinysim.scene.element import Element
from tinysim.core.transform import Rotation

from tinysim.core.renderer import SimulationRenderer as Renderer


def simulate(env : Element, **kwargs):
  return Simulation(scene=env, **kwargs)

class Simulation:

  def __init__(self, scene : Element = None, renderer = "mjviewer", visualize_groups = set(range(3)), render_args = {}):

    self.model = None
    self.visualize_groups = visualize_groups   
    self.renderer : Renderer = Renderer.create(renderer, **render_args)

    if scene is not None:
      self.load_environment(scene)


  def load_environment(self, scene : Element):
  
    self.model = scene.compile()
    self.env = scene
    self.joints = scene.joints
    self.objects = scene.bodies


    self.data = mj.MjData(self.model)
    mj.mj_forward(self.model, self.data)

    self.env._on_simulation_init(self)
    self._scene_update()


    self.renderer.init_scene(self)
    self.renderer.update_scene(self)

  def close(self):
    self.renderer.close()

  def step(self):
    
    self.env.step()

    mj.mj_step(self.model, self.data)
    
    self._scene_update()
    self.renderer.update_scene(self)

  def get_renderer(self) -> Renderer:
    return self.renderer

  def is_running(self):
    return self.renderer.is_running()

  def _scene_update(self):

    # update sim bodies pose
    for obj in self.objects:
      obj.xpos = torch.from_numpy(self.data.xpos[obj.id].copy())
      obj.xrot = Rotation(self.data.xquat[obj.id][[1, 2, 3, 0]].copy())

    for joint in self.joints:
      joint.qpos = self.data.qpos[self.model.jnt(joint.id).qposadr].copy()
      joint.qvel = self.data.qvel[self.model.jnt(joint.id).qposadr].copy()