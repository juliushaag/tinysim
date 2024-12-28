from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

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



class Renderer(ABC):

  BACKENDS = dict()

  @classmethod
  def register_backend(cls, name : str, backend : "Renderer"):
    cls.BACKENDS[name] = backend

  @classmethod
  def create(cls, name : str, **kwargs) -> "Renderer":
    return cls.BACKENDS[name](**kwargs)

  def __init__(self):
    pass

  @abstractmethod
  def create_mesh(self, name : str, vertices: np.ndarray, indices : np.ndarray, normals: np.ndarray, uvs: np.ndarray = None) -> str:
    pass

  @abstractmethod
  def create_material(self, name : str, color : np.ndarray, emission : float, specular : float, shininess : float, reflectance : float, texture : str = None) -> str:
    pass

  @abstractmethod
  def create_texture(self, name : str, data : np.ndarray, width : int, height : int) -> str:
    pass

  @abstractmethod
  def create_object(self, name : str, transform : RenderTransform, parent : str) -> str:
    pass

  @abstractmethod
  def attach_mesh(self, obj_name : str, mesh : str, material : str, transform : RenderTransform) -> str:
    pass

  @abstractmethod
  def attach_primitive(self, obj_name : RenderPrimitiveType, type : str, material : str, transform : RenderTransform) -> str:
    pass

  @abstractmethod
  def update_transform(self, name : str, transform : RenderTransform) -> None:
    pass

  @abstractmethod
  def close(self):
    ...
