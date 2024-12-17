import math
import time
import mujoco as mj

from tinysim.core.scene import Scene, SceneVisualType
from tinysim.renderer import RenderTransform, Renderer

from scipy.spatial.transform import Rotation as R

def simulate(scene : Scene, **kwargs):
  return Simulation(scene, **kwargs)

class Simulation:

  def __init__(self, scene : Scene, renderer = "web", render_args = {}):
    scene.compile()
    self.scene = scene
    self.mj_data = mj.MjData(scene.mj_model)
    mj.mj_setKeyframe(scene.mj_model, self.mj_data, 0)
    mj.mj_forward(scene.mj_model, self.mj_data)

    self.renderer = Renderer.create(renderer, **render_args)
    self._render_scene()


  def step(self):
    mj.mj_step(self.scene.mj_model, self.mj_data)
    # self.scene.update(self.mj_data)
    self._render_update()
    self.renderer.update()
    

  def _render_scene(self):

    for mesh in self.scene.meshes:
      self.renderer.create_mesh(mesh.name, mesh.vertices, mesh.indices, mesh.normals, mesh.uvs)
    
    for mesh in self.scene.materials:
      self.renderer.create_material(mesh.name, mesh.color, mesh.emission, mesh.specular, mesh.shininess, mesh.reflectance, mesh.texture)

    for mesh in self.scene.textures:
      self.renderer.create_texture(mesh.name, mesh.data, mesh.width, mesh.height)

    objs = [(None, self.scene.root)]
    self.tracked_objs = list()
    while len(objs) > 0:
      parent_name, obj = objs.pop(0)

      if (obj.movable):
        self.tracked_objs.append(obj)

      self.renderer.create_object(obj.name, RenderTransform(obj.position, obj.quaternion), parent_name)

      # All meshes are z-up we need y-up, but still right hand coord system ...
      visuals = self.renderer.create_object(None, RenderTransform(quaternion=R.from_euler("xyz", [math.pi, 0.0, -math.pi / 2]).as_quat()), obj.name)

      for visual in obj.visuals:
        if visual.group == 3: continue
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

  def _render_update(self):
    for obj in self.tracked_objs:
      if obj.movable:
        self.renderer.update_transform(obj.name, RenderTransform(obj.position, obj.quaternion))
