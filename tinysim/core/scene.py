from dataclasses import dataclass
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
import trimesh
from tinysim.core.robot import Robot

import mujoco as mj

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict
from enum import Enum
import json
from hashlib import md5

class SceneVisualType(str, Enum):
  CUBE = "CUBE"
  SPHERE = "SPHERE"
  CAPSULE = "CAPSULE"
  CYLINDER = "CYLINDER"
  PLANE = "PLANE"
  QUAD = "QUAD"
  MESH = "MESH"
  NONE = "NONE"


@dataclass
class SceneMaterial:
  name: str
  color: List[float]
  emission: float = 0.5
  specular: float = 0.5
  shininess: float = 0.5
  reflectance: float = 0.0
  texture: Optional[str] = None
  texrepeat: np.ndarray = field(default_factory=lambda: np.array([1, 1]))


@dataclass
class SceneMesh:
    name: str
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray


@dataclass
class SceneTexture:
    name: str
    data: np.ndarray
    width: int = 0
    height: int = 0
    textureType: str = "2D"
    textureSize: Tuple[int, int] = field(default_factory=lambda: (1, 1))


@dataclass
class SceneVisual:
    name: str
    group : int
    type: SceneVisualType
    color: list[float]
    position: List[float] = field(default_factory=lambda: np.zeros(3))
    quaternion: List[float] = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    scale: List[float] = field(default_factory=lambda: np.ones(3))
    material: Optional[str] = None
    mesh: Optional[str] = None

@dataclass
class SceneObject:
    name: str
    id : int = -1
    movable: bool = False
    position: List[float] = field(default_factory=lambda: np.zeros(3))
    quaternion: List[float] = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    visuals: List[SceneVisual] = field(default_factory=lambda: list())
    children: List["SceneObject"] = field(default_factory=lambda: list())

MJ2TINYSIM = {
  mj.mjtGeom.mjGEOM_SPHERE: SceneVisualType.SPHERE,
  mj.mjtGeom.mjGEOM_CAPSULE: SceneVisualType.CAPSULE,
  mj.mjtGeom.mjGEOM_ELLIPSOID: SceneVisualType.CAPSULE,
  mj.mjtGeom.mjGEOM_CYLINDER: SceneVisualType.CYLINDER,
  mj.mjtGeom.mjGEOM_BOX: SceneVisualType.CUBE,
  mj.mjtGeom.mjGEOM_MESH: SceneVisualType.MESH,
  mj.mjtGeom.mjGEOM_PLANE: SceneVisualType.PLANE,
}


SCENES_PATH = (Path(__file__).parent / "../../models/scenes").resolve()
 
SCENES = { path.name : path for path in SCENES_PATH.iterdir() }


@dataclass
class SceneConfig:
  definition: str
  robot_mount_points: Union[str, list]

def load_scene(name : str) -> "Scene":
  return Scene.load(name)

class Scene:

  @classmethod
  def load(cls, scene_name : str) -> "Scene":
    
    if scene_name not in SCENES:
      raise ValueError("Invalid scene, select one of", SCENES.keys())
    
    conf = SCENES[scene_name] / "description.yaml"

    if not conf.is_file():
      raise ValueError("No 'description.yaml' found for", scene_name)

    scene_config = SceneConfig(**yaml.full_load(conf.read_text()))

    scene_config.definition = str(SCENES[scene_name] / scene_config.definition)

    return cls(scene_config)
  

  def __init__(self, conf : SceneConfig) -> None:

    self.conf = conf    
    self.xml_file = Path(conf.definition)
    self.xml_spec = mj.MjSpec.from_file(str(self.xml_file))

    self.id = None
    self.root = None

    mount_points = self.conf.robot_mount_points if isinstance(self.conf.robot_mount_points, list) else [self.conf.robot_mount_points]

    self.robots = []
    self.mount_points : dict[str, Optional[Robot]]= { name : None for name in mount_points }
    self.compiled = False

  def attach(self, robot : Robot, mount_point = None):

    if mount_point:
      assert self.mount_points[mount_point] == None, f"Mount point {mount_point} is already occupied"
      self.mount_points[mount_point] = robot
    else:
      free_mounts = [key for key in self.mount_points.keys() if self.mount_points[key] is None]
      assert len(free_mounts) > 0, "Not more free mounts to attach to"
      self.mount_points[free_mounts[0]] = robot
    
  def compile(self):

    scene_spec = self.xml_spec

    for mount_point, robot in self.mount_points.items():
      if robot is None: continue
      mp = scene_spec.find_body(mount_point)
      
      assert mp, "Mount point not in scene"

      robot_frame = mp.add_frame()
      xml_robot = mj.MjSpec.from_file(str(robot.get_xml_file()))
      robot_frame.attach_body(xml_robot.worldbody, robot.name, '')


    self.mj_model = scene_spec.compile()
    self.compiled = True
    self.id: str = md5(scene_spec.to_xml().encode()).hexdigest()

    
    self.root = self.load_bodys(self.mj_model)

    self.materials = [self.load_material(self.mj_model.material(i)) for i in range(self.mj_model.nmat)]    
    self.meshes = [self.load_mesh(self.mj_model.mesh(i), self.mj_model) for i in range(self.mj_model.nmesh)]
    self.textures = [self.load_texture(self.mj_model.tex(i), self.mj_model, i) for i in range(self.mj_model.ntex)]
    


  def load_bodys(self, model : mj.MjModel): 

    bodies = {}
    for body_id in range(model.nbody):
      body = model.body(body_id)
      
      bod = SceneObject(
        name=body.name,
        position=body.pos, 
        quaternion=body.quat,
        movable = len(body.jntnum) > 0,
        id = body_id
      )

      bodies[body_id] = bod
      if body_id != body.parentid.item():
        bodies[body.parentid.item()].children.append(bod)

      for geom_id in range(body.geomadr.item(), body.geomadr.item() + body.geomnum.item()):

        geom = model.geom(geom_id)
        type = MJ2TINYSIM[geom.type.item()]
        
        visual = SceneVisual(
          name=geom.name,
          type=type,
          color=geom.rgba,
          group=geom.group,
          position=geom.pos,
          quaternion=geom.quat, 
          scale=geom.size if type != SceneVisualType.MESH else [1, 1, 1],
          material= None if geom.matid == -1 else model.material(geom.matid).name,
          mesh= None if geom.dataid == -1 else model.mesh(geom.dataid).name
        )

        bod.visuals.append(visual)

    return bodies[0]

  def load_texture(self, texture, model, id) -> SceneTexture:

    height = texture.height.item()
    width = texture.width.item()
    
    start_tex = model.tex_adr[id].item()
    size = height * width * 3
    
    return SceneTexture(
      name=texture.name,
      data=np.copy(model.tex_data[start_tex:start_tex + size]).reshape((height, width, 3)),
      width=width,
      height=height,
      textureType="2D",
    )

  def load_material(self, material) -> SceneMaterial:
    return SceneMaterial(
        name=material.name,
        color=material.rgba,
        emission=material.emission.item(),
        specular=material.specular.item(),
        shininess=material.shininess.item(),
        reflectance=material.reflectance.item(),
        texture=None,
    )
     

  def load_mesh(self, mesh, model) -> SceneMesh:
    

    # vertices
    start_vert = mesh.vertadr.item()
    num_verts = mesh.vertnum.item()

    vertices = model.mesh_vert[start_vert:start_vert + num_verts]
    vertices = vertices.copy().astype(np.float32)
    
    # normal 
    normals = model.mesh_normal[start_vert:start_vert + num_verts]
    normals = normals.copy().astype(np.float32)

    # faces
    start_face = mesh.faceadr.item()
    num_faces = mesh.facenum.item()

    faces = model.mesh_face[start_face:start_face + num_faces]
    indices = faces.copy().astype(np.int32)

    # Texture coords
    uvs =  None
    start_uv = mesh.texcoordadr.item()
    if start_uv != -1:
      num_texcoord = mesh.texcoordnum.item()
      uvs = model.mesh_texcoord[start_uv:start_uv + num_texcoord]

    return SceneMesh(
      name=mesh.name,
      indices=indices,
      vertices=vertices,
      normals=normals,
      uvs=uvs,
    )