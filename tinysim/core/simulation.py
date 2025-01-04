import time
import math
import mujoco as mj
import numpy as np

from tinysim.scene.environment import Environment
from scipy.spatial.transform import Rotation as R

from tinysim.core.renderer import SimulationRenderer as Renderer


def simulate(env : Environment, **kwargs):
  return Simulation(env=env, **kwargs)

class Simulation:

  def __init__(self, env : Environment = None, renderer = "mjviewer", visualize_groups = set(range(3)), render_args = {}):

    self.model = None
    self.visualize_groups = visualize_groups   
    self.renderer : Renderer = Renderer.create(renderer, **render_args)

    if env is not None:
      self.load_environment(env)

  def load_environment(self, env : Environment):
  
    self.model = env.compile()
    self.env = env
    self.joints = env.joints
    self.objects = env.bodies


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
    for obj in self.objects:
      parent_id = self.model.body_parentid[obj.id]

      # Get world positions
      obj.position = body_pos = self.data.xpos[obj.id]
      obj.quaternion = body_quat = self.data.xquat[obj.id]

      parent_pos = self.data.xpos[parent_id]

      # Get inv of parent's rotation matrix
      parent_rot = R.from_matrix(self.data.xmat[parent_id].reshape(3, 3)).inv()
      
      # Calculate relative position in parent's frame
      rel_pos = body_pos - parent_pos

      obj.position_rel = parent_rot.apply(rel_pos)
      obj.quaternion_rel = (parent_rot * R.from_quat(body_quat, scalar_first=True)).as_quat(scalar_first=True)
    
    for joint in self.joints:
      joint.qpos = self.data.qpos[self.model.jnt(joint.id).qposadr]
      joint.qvel = self.data.qvel[self.model.jnt(joint.id).qposadr]
    
