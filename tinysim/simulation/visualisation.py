from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import mujoco as mj

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
        id=None,
        type=SceneVisualType.from_mj(spec.type),
        color=spec.rgba,
        group=spec.group,
        position=spec.pos,
        quaternion=spec.quat,
        scale=spec.size if spec.type != mj.mjtGeom.mjGEOM_MESH else [1, 1, 1],
        material = spec.material,
        mesh = spec.meshname
      )
