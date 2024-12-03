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


@dataclass
class SceneMesh:
    name: str
    hash : str
    verticesLayout: Tuple[int, int]
    indicesLayout: Tuple[int, int]
    uvLayout: Tuple[int, int]
    normalsLayout: Tuple[int, int]


@dataclass
class SceneTexture:
    name: str
    hash: str
    width: int = 0
    height: int = 0
    textureType: str = "2D"
    textureSize: Tuple[int, int] = field(default_factory=lambda: (1, 1))


@dataclass
class SceneTransform:
    pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    rot: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    scale: List[float] = field(default_factory=lambda: [1, 1, 1])

@dataclass
class SceneVisual:
    name: str
    type: SceneVisualType
    trans: SceneTransform
    color: list[float]
    material: Optional[str] = None
    mesh: Optional[str] = None

@dataclass
class SceneObject:
    name: str
    group: int
    trans: SceneTransform = field(default_factory=SceneTransform)
    visuals: List[SceneVisual] = field(default_factory=list)
    children: List["SceneObject"] = field(default_factory=list)

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


  def attach(self, robot : Robot, mount_point = None):

    if mount_point:
      assert self.mount_points[mount_point] == None, f"Mount point {mount_point} is already occupied"
      self.mount_points[mount_point] = robot
    else:
      free_mounts = [key for key in self.mount_points.keys() if self.mount_points[key] is None]
      assert len(free_mounts) > 0, "Not more free mounts to attach to"
      self.mount_points[free_mounts[0]] = robot
    
  def compile(self) -> mj.MjModel:

    scene_spec = self.xml_spec

    for mount_point, robot in self.mount_points.items():
      if robot is None: continue
      mp = scene_spec.find_body(mount_point)
      
      assert mp, "Mount point not in scene"

      robot_frame = mp.add_frame()
      xml_robot = mj.MjSpec.from_file(str(robot.get_xml_file()))
      robot_frame.attach_body(xml_robot.worldbody, robot.name, '')


    model = scene_spec.compile()
    self.id: str = md5(scene_spec.to_xml().encode()).hexdigest()

    self.root = SceneObject("world", group=0) 
    root = scene_spec.worldbody 

    self.load_bodys(root, self.root, model)

    self.load_assets(model)
    
    return model


  def load_bodys(self, parent_spec: mj.MjSpec, body: SceneObject, model : mj.MjModel): 

    for geom_spec in parent_spec.geoms:
      
      type = MJ2TINYSIM[geom_spec.type]

      visual = SceneVisual(
        name=geom_spec.name,
        type=type,
        trans=SceneTransform(model.geom_pos[geom_spec.id], model.geom_quat[geom_spec.id], geom_spec.size if type != SceneVisualType.MESH else [1, 1, 1]),
        material=geom_spec.material,
        color=geom_spec.rgba,
        mesh= None if geom_spec.meshname == "" else geom_spec.meshname
      )
      body.visuals.append(visual)
      
    
    for body_spec in parent_spec.bodies:
      child = SceneObject(
        name=body_spec.name,
        trans=SceneTransform(model.body_pos[body_spec.id], model.body_quat[body_spec.id]),
        group=0
      )

      body.children.append(child)

      self.load_bodys(body_spec, child, model)

  def load_assets(self, model : mj.MjModel):
    
    self.assets = dict()
    self.meshes = list()
    self.textures = list()
    self.materials = list()

    for i in range(model.nmesh):
      mesh = model.mesh(i)
      scene_mesh, data = self.load_mesh(mesh, model)
      self.assets[scene_mesh.hash] = data
      self.meshes.append(scene_mesh)


    for i in range(model.nmat):
      material = self.load_material(model.material(i))
      self.materials.append(material)

    
    for i in range(model.ntex):
      texture = model.tex(i)
      scene_texture, data = self.load_texture(texture, model, i)
      self.textures.append(scene_texture)
      self.assets[scene_texture.hash] = data

  def load_texture(self, texture, model, id) -> tuple[SceneTexture, bytes]:

    height = texture.height.item()
    width = texture.width.item()
    
    start_tex = model.tex_adr[id].item()
    size = height * width * 3
    
    data: np.ndarray = model.tex_rgb[start_tex:start_tex + size]

    bin_data = data.astype(np.uint8).tobytes()
    
    texture_hash = md5(bin_data).hexdigest()
    texture = SceneTexture(
        hash=texture_hash,
        width=width,
        height=height,
        textureType="2D",
    )
    
    return texture, bin_data
    

  def load_material(self, material) -> SceneMaterial:
    return SceneMaterial(
        name=material.name,
        color=material.rgba,
        emission=material.emission,
        specular=material.specular,
        shininess=material.shininess,
        reflectance=material.reflectance,
        texture=None,
    )
     

  def load_mesh(self, mesh, model) -> tuple[SceneMesh, bytes]:
    
    bin_buffer = io.BytesIO()

    # vertices
    start_vert = mesh.vertadr.item()
    num_verts = mesh.vertnum.item()
    vertices = model.mesh_vert[start_vert:start_vert + num_verts]
    
    vertices = vertices.copy().astype(np.float32).flatten()
    vertices_layout = bin_buffer.tell(), vertices.shape[0]
    bin_buffer.write(vertices)
    
    # normal 
    norms = model.mesh_normal[start_vert:start_vert + num_verts]

    norms = norms.copy().astype(np.float32).flatten()
    normal_layout = bin_buffer.tell(), norms.shape[0]
    bin_buffer.write(norms)

    # faces
    start_face = mesh.faceadr.item()
    num_faces = mesh.facenum.item()
    faces = model.mesh_face[start_face:start_face + num_faces]

    indices = faces.copy().astype(np.int32).flatten()
    indices_layout = bin_buffer.tell(), indices.shape[0]
    bin_buffer.write(indices)

    # Texture coords
    uv_layout = (0, 0)
    start_uv = mesh.texcoordadr.item()
    if start_uv != -1:
        num_texcoord = mesh.texcoordnum.item()

        uvs = model.mesh_texcoord[start_uv:start_uv + num_texcoord].copy().flatten()
        uv_layout = bin_buffer.tell(), uvs.shape[0]
        bin_buffer.write(uvs)
        
    # create a SiMmesh object and raw data
    bin_data = bin_buffer.getvalue()
    hash = md5(bin_data).hexdigest()

    mesh = SceneMesh(
      name=mesh.name,
      indicesLayout=indices_layout,
      verticesLayout=vertices_layout,
      normalsLayout=normal_layout,
      uvLayout=uv_layout,
      hash=hash
    )
    return mesh, bin_data