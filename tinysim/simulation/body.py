
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from tinysim.core.transform import Rotation
from tinysim.simulation.joint import Joint

import mujoco as mj

@dataclass
class SceneBody:
  name: str
  id : int = -1
  movable: bool = False

  xpos: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
  xrot: Rotation = field(default_factory=Rotation)

  ipos : torch.Tensor = field(default_factory=lambda: torch.zeros(3))
  irot : Rotation  = field(default_factory=Rotation)

  joints : list[Joint] = field(default_factory=lambda: list())
  children: list["SceneBody"] = field(default_factory=lambda: list())

  spec : mj.MjsBody = None
  parent : "SceneBody" = None

  @classmethod
  def from_spec(cls, spec, parent = None):
    print(spec.quat.copy()[[1, 2, 3, 0]])
    body =  cls(
      name=spec.name,


      ipos=torch.from_numpy(spec.pos.copy()), 
      irot=Rotation(spec.quat.copy()[[1, 2, 3, 0]]),

      movable = len(spec.joints) > 0,
      parent=parent,
      id=None,
      spec=spec,

    )
    body.children = [SceneBody.from_spec(child, body) for child in spec.bodies]
    body.joints = [Joint.from_spec(joint) for joint in spec.joints]
    
    return body
  
  def attach(self, body : "SceneBody", namespace : str) -> "SceneBody":
    frame = self.spec.add_frame()
    frame.attach_body(body.spec, namespace, '')
    self.children.append(body)

  def get_all_bodies(self) -> List["SceneBody"]:
    bodies = [self]
    for child in self.children:
      bodies.extend(child.get_all_bodies())
    return bodies
  
  def get_all_joints(self) -> List[Joint]:
    joints = list()
    for child in self.get_all_bodies():
      joints.extend(child.joints)
    return joints
  
  @property
  def rpos(self) -> np.ndarray:
    if not self.movable: return self.ipos
    if not self.parent: return self.xpos

    return self.parent.xrot.inv().rotate(self.xpos - self.parent.rpos)
  
  @property
  def rrot(self) -> Rotation:
    if not self.movable: return self.irot
    if not self.parent: return self.xrot

    return self.xrot * self.parent.rrot.inv()

  def __repr__(self):
    return f"<SceneBody {self.name} children=[{",".join(obj.name for obj in self.children)}]>"