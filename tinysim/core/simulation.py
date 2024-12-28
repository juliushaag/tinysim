import time
import math
import mujoco as mj

from tinysim.scene.environment import Environment
from tinysim.scene.scene import SceneElement, SceneMaterial, SceneMesh, SceneTexture, SceneVisualType, SceneBody
from tinysim.renderer import RenderTransform, Renderer

from scipy.spatial.transform import Rotation as R


def simulate(env : Environment, **kwargs):
  return Simulation(env=env, **kwargs)

class Simulation:

  def __init__(self, env : Environment = None, renderer = "web", visualize_groups = set(range(10)), render_args = {}):

    self.model = None
    self.visualize_groups = visualize_groups   
    self.renderer = Renderer.create(renderer, **render_args)

    if env is not None:
      self.load_environment(env)

  def load_environment(self, env : Environment):
  
    self.model = env._compile()

    self.meshes = SceneMesh.from_model(self.model)
    self.textures = SceneTexture.from_model(self.model)
    self.materials = SceneMaterial.from_model(self.model)


    self.env = env
    self.joints = env.joints

    self.data = mj.MjData(self.model)
    mj.mj_forward(self.model, self.data)

    self._render_scene()
    self._scene_update()

  def close(self):
    self.renderer.close()

  def step(self):

    mj.mj_step(self.model, self.data)
    
    self._scene_update()
    self.renderer.update()

  def get_renderer(self) -> Renderer:
    return self.renderer

  def _render_scene(self):

    for mesh in self.meshes:
      self.renderer.create_mesh(mesh.name, mesh.vertices, mesh.indices, mesh.normals, mesh.uvs)
    
    for mesh in self.materials:
      self.renderer.create_material(mesh.name, mesh.color, mesh.emission, mesh.specular, mesh.shininess, mesh.reflectance, mesh.texture)

    for mesh in self.textures:
      self.renderer.create_texture(mesh.name, mesh.data, mesh.width, mesh.height)

    objs = [(None, self.env.root)]
    self.tracked_objs = list()
    while len(objs) > 0:
      parent_name, obj = objs.pop(0)


      if (obj.movable):
        self.tracked_objs.append(obj)

      self.renderer.create_object(obj.name, RenderTransform(obj.position, obj.quaternion), parent_name)

      # All meshes are z-up we need y-up, but still right hand coord system ...
      visuals  = self.renderer.create_object(None, RenderTransform(quaternion=R.from_euler("x", -math.pi / 2).as_quat(scalar_first=True)), obj.name)

      for visual in obj.visuals:
        if visual.group.item() in self.visualize_groups: continue
        if visual.type == SceneVisualType.MESH:
          self.renderer.attach_mesh(
            obj_name=visuals,
            mesh=visual.mesh, 
            material=visual.material or self.renderer.create_material(None, visual.color), 
            transform=RenderTransform(visual.position, visual.quaternion)
          )
        else:
          self.renderer.attach_primitive(
            obj_name=obj.name, 
            type=visual.type, 
            material=visual.material or self.renderer.create_material(None, visual.color),
            transform=RenderTransform(visual.position, visual.quaternion, visual.scale), 
          )

      objs.extend((obj.name, child) for child in obj.children)

  def _scene_update(self):
    for obj in self.tracked_objs:
      parent_id = self.model.body_parentid[obj.id]

      # Get world positions
      body_pos = self.data.xpos[obj.id]
      parent_pos = self.data.xpos[parent_id]

      body_quat = self.data.xquat[obj.id]
      
      # Get parent's rotation matrix
      parent_rot = R.from_matrix(self.data.xmat[parent_id].reshape(3, 3)).inv()
      
      # Calculate relative position in parent's frame
      rel_pos = body_pos - parent_pos

      obj.position = parent_rot.apply(rel_pos)
      obj.quaternion = (parent_rot * R.from_quat(body_quat, scalar_first=True)).as_quat(scalar_first=True)

      self.renderer.update_transform(obj.name, RenderTransform(obj.position, obj.quaternion))
    
    for joint in self.joints:
      joint.qpos = self.data.qpos[joint.id]
      joint.qvel = self.data.qvel[joint.id]
    
