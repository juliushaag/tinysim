from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import mujoco as mj

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

class SceneVisualType(str, Enum):
  CUBE = "CUBE"
  SPHERE = "SPHERE"
  CAPSULE = "CAPSULE"
  CYLINDER = "CYLINDER"
  PLANE = "PLANE"
  QUAD = "QUAD"
  MESH = "MESH"
  NONE = "NONE"

  @staticmethod
  def from_mj(mj_type):
    return {
      mj.mjtGeom.mjGEOM_SPHERE: SceneVisualType.SPHERE,
      mj.mjtGeom.mjGEOM_CAPSULE: SceneVisualType.CAPSULE,
      mj.mjtGeom.mjGEOM_ELLIPSOID: SceneVisualType.CAPSULE,
      mj.mjtGeom.mjGEOM_CYLINDER: SceneVisualType.CYLINDER,
      mj.mjtGeom.mjGEOM_BOX: SceneVisualType.CUBE,
      mj.mjtGeom.mjGEOM_MESH: SceneVisualType.MESH,
      mj.mjtGeom.mjGEOM_PLANE: SceneVisualType.PLANE,
    } [mj_type]

class SceneJointType(str, Enum):
  FREE = "FREE",
  BALL = "BALL",
  SLIDE = "SLIDE",
  HINGE = "HINGE",

  @staticmethod
  def from_mj(mj_type):
    return {
      mj.mjtJoint.mjJNT_FREE: SceneJointType.FREE,
      mj.mjtJoint.mjJNT_BALL: SceneJointType.BALL,
      mj.mjtJoint.mjJNT_SLIDE: SceneJointType.SLIDE,
      mj.mjtJoint.mjJNT_HINGE: SceneJointType.HINGE,
    } [mj_type]

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

  @classmethod
  def from_model(cls, model) -> list["SceneMaterial"]:
    materials = list()
    for mat_id in range(model.nmat):
      material = model.mat(mat_id)
      materials.append(cls(
        name=material.name,
        color=material.rgba,
        emission=material.emission,
        specular=material.specular,
        shininess=material.shininess,
        reflectance=material.reflectance,
        # texture=material.texture,
        texrepeat=material.texrepeat,
      ))
    return materials

@dataclass(frozen=True)
class SceneMesh:
  name: str
  vertices: np.ndarray
  indices: np.ndarray
  normals: np.ndarray
  uvs: np.ndarray

  @classmethod
  def from_model(cls, model) -> list["SceneMesh"]: 

    meshes = list()
    for mesh_id in range(model.nmesh):
      mesh = model.mesh(mesh_id)
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
      # start_uv = mesh.texcoordadr.item()
      # if start_uv != -1:
      #   num_texcoord = mesh.texcoordnum.item()
      #   uvs = model.mesh_texcoord[start_uv:start_uv + num_texcoord]

      meshes.append(cls(
        name=mesh.name,
        indices=indices,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
      ))
    return meshes

@dataclass
class SceneTexture:
  name: str
  data: np.ndarray
  width: int = 0
  height: int = 0
  textureType: str = "2D"
  textureSize: Tuple[int, int] = field(default_factory=lambda: (1, 1))

  @classmethod
  def from_model(cls, model) -> list["SceneTexture"]:
    textures = list()
    for tex_id in range(model.ntex):
      texture = model.tex(tex_id)
      height = texture.height.item()
      width = texture.width.item()
      
      start_tex = model.tex_adr[tex_id].item()
      size = height * width * 3
      
      textures.append(cls(
        name=texture.name,
        data=np.copy(model.tex_data[start_tex:start_tex + size]).reshape((height, width, 3)),
        width=width,
        height=height,
        textureType="2D",
      ))
    return textures


@dataclass
class SceneVisual:
    name: str
    id : int
    group : int
    type: SceneVisualType
    color: list[float]
    position: List[float] = field(default_factory=lambda: np.zeros(3))
    quaternion: List[float] = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    scale: List[float] = field(default_factory=lambda: np.ones(3))
    material: Optional[str] = None
    mesh: Optional[str] = None

    @classmethod
    def from_spec(cls, spec):
      return cls(
        name=spec.name,
        id=spec.id,
        type=SceneVisualType.from_mj(spec.type),
        color=spec.rgba,
        group=spec.group,
        position=spec.pos,
        quaternion=spec.quat,
        scale=spec.size if spec.type != mj.mjtGeom.mjGEOM_MESH else [1, 1, 1],
        material = spec.material,
        mesh = spec.meshname
      )
    
    @classmethod
    def from_model(cls, geom, model):
      type = SceneVisualType.from_mj(geom.type.item())
      return  SceneVisual(
        name=geom.name,
        id=geom.id,
        type=type,
        color=geom.rgba,
        group=geom.group,
        position=geom.pos,
        quaternion=geom.quat, 
        scale=geom.size if type != SceneVisualType.MESH else [1, 1, 1],
        material= None if geom.matid == -1 else model.material(geom.matid).name,
        mesh= None if geom.dataid == -1 else model.mesh(geom.dataid).name
      )
    
@dataclass
class SceneJoint:
  name : str
  id : int
  type : SceneJointType
  pos : np.ndarray
  axis : np.ndarray
  range : Tuple[float, float]
  qpos : np.ndarray = field(default_factory=lambda: np.zeros(1))
  qvel : np.ndarray = field(default_factory=lambda: np.zeros(1))

  @classmethod
  def from_spec(cls, spec):
    return cls(
      name=spec.name,
      id =spec.id,
      type=SceneJointType.from_mj(spec.type),
      pos=spec.pos,
      axis=spec.axis,
      range=(spec.range[0], spec.range[1])
    )
  
  @classmethod
  def from_model(cls, joint, model):
    return cls(
      name=joint.name,
      id = joint.id,
      type=SceneJointType.from_mj(joint.type.item()),
      pos=joint.pos,
      axis=joint.axis,
      range=(joint.range[0], joint.range[1]),
      qpos=model.qpos0[joint.qposadr],
      qvel=np.zeros(1)
    )

  def __repr__(self):
    return f"<SceneJoint {self.name} type={self.type} range=[{self.range[0].item()}:{self.range[1].item()}]>"

@dataclass
class SceneBody:
  name: str
  id : int = -1
  movable: bool = False
  position: List[float] = field(default_factory=lambda: np.zeros(3))
  quaternion: List[float] = field(default_factory=lambda: np.array([1, 0, 0, 0]))
  visuals: List[SceneVisual] = field(default_factory=lambda: list())
  joints : List[SceneJoint] = field(default_factory=lambda: list())
  children: List["SceneBody"] = field(default_factory=lambda: list())
  spec : mj.MjsBody = None

  @classmethod
  def from_spec(cls, spec):
      return cls(
      name=spec.name,
      position=spec.pos, 
      quaternion=spec.quat,
      movable = len(spec.joints) > 0,
      id=spec.id,
      spec=spec,
      visuals = [SceneVisual.from_spec(geom) for geom in spec.geoms],
      children = [SceneBody.from_spec(child) for child in spec.bodies],
      joints = [SceneJoint.from_spec(joint) for joint in spec.joints]
    )
  
  @classmethod
  def from_model(cls, model) -> "SceneBody":
    bodies = dict()
    for body_id in range(model.nbody):
      body = model.body(body_id)
      
      bod = cls(
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
        bod.visuals.append(SceneVisual.from_model(geom, model))

      for joint_id in range(body.jntadr.item(), body.jntadr.item() + body.jntnum.item()):
        joint = model.joint(joint_id)
        bod.joints.append(SceneJoint.from_model(joint, model))

    return bodies[0]
  
  def attach(self, body : "SceneBody", namespace : str) -> "SceneBody":
    frame = self.spec.add_frame()
    new_spec = frame.attach_body(body.spec, namespace, '')
    return SceneBody.from_spec(new_spec)
  

  def get_all_bodies(self) -> List["SceneBody"]:
    bodies = [self]
    for child in self.children:
      bodies.extend(child.get_all_bodies())
    return bodies
  
  def get_all_joints(self) -> List[SceneJoint]:
    joints = self.joints
    for child in self.children:
      joints.extend(child.get_all_joints())
    return joints

  def __repr__(self):
    return f"<SceneBody {self.name} children=[{",".join(obj.name for obj in self.children)}]>"


class SceneElement:
  def __init__(self, name : str, spec) -> None:
    self._name = name
    self._position = np.zeros(3)
    self._quaternion = np.zeros(4)
    self._spec = spec
    self._bodies = dict()
    self._joints = dict()
    self._visuals = dict()

    self._reload(SceneBody.from_spec(spec.worldbody))

  @property
  def name(self):
    return self._name
  
  @property
  def root(self):
    return self._root
  
  @property
  def bodies(self):
    return list(self._bodies.values())
  
  @property
  def joints(self):
    return list(self._joints.values())
  
  def get_body_by_name(self, name : str) -> SceneBody:
    return self._bodies[name]
  
  def get_joint_by_name(self, name : str) -> SceneBody:
    return self._joints[name]
  
  def _on_simulation_init(self):
    pass
  
  def _reload(self, root : SceneBody):
    self._root = root
    self._bodies = { body.name : body for body in root.get_all_bodies() }
    self._joints = { joint.name : joint for joint in root.get_all_joints() }
    self._visuals = { visual.name : visual for visual in root.visuals }
