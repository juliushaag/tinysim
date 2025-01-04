
from dataclasses import dataclass, field
from typing import List

import numpy as np

from tinysim.simulation.visualisation import SceneVisual
from tinysim.simulation.joint import SceneJoint

import mujoco as mj

@dataclass
class SceneBody:
  name: str
  id : int = -1
  movable: bool = False
  position: List[float] = field(default_factory=lambda: np.zeros(3))
  quaternion: List[float] = field(default_factory=lambda: np.array([1, 0, 0, 0]))
  position_rel : List[float] = field(default_factory=lambda: np.zeros(3))
  quaternion_rel : List[float] = field(default_factory=lambda: np.array([1, 0, 0, 0]))
  visuals: List[SceneVisual] = field(default_factory=lambda: list())
  joints : List[SceneJoint] = field(default_factory=lambda: list())
  children: List["SceneBody"] = field(default_factory=lambda: list())
  spec : mj.MjsBody = None

  @classmethod
  def from_spec(cls, spec):
    return cls(
      name=spec.name,
      position_rel=spec.pos, 
      quaternion_rel=spec.quat,
      movable = len(spec.joints) > 0,
      id=None,
      spec=spec,
      visuals = [SceneVisual.from_spec(geom) for geom in spec.geoms],
      children = [SceneBody.from_spec(child) for child in spec.bodies],
      joints = [SceneJoint.from_spec(joint) for joint in spec.joints],
    )
  
  def attach(self, body : "SceneBody", namespace : str) -> "SceneBody":
    frame = self.spec.add_frame()
    frame.attach_body(body.spec, namespace, '')
    self.children.append(body)

  def get_all_bodies(self) -> List["SceneBody"]:
    bodies = [self]
    for child in self.children:
      bodies.extend(child.get_all_bodies())
    return bodies
  
  def get_all_joints(self) -> List[SceneJoint]:
    joints = list()
    for child in self.get_all_bodies():
      joints.extend(child.joints)
    return joints

  def __repr__(self):
    return f"<SceneBody {self.name} children=[{",".join(obj.name for obj in self.children)}]>"