from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
import mujoco as mj
import torch

from tinysim.core.transform import Rotation, Transform

class JointType(str, Enum):
  FREE = "FREE",
  BALL = "BALL",
  SLIDE = "SLIDE",
  HINGE = "HINGE",

  @staticmethod
  def from_mj(mj_type):
    return {
      mj.mjtJoint.mjJNT_FREE: JointType.FREE,
      mj.mjtJoint.mjJNT_BALL: JointType.BALL,
      mj.mjtJoint.mjJNT_SLIDE: JointType.SLIDE,
      mj.mjtJoint.mjJNT_HINGE: JointType.HINGE,
    } [mj_type]
  
@dataclass
class Joint(ABC):
  id : int
  name : str
  qpos : np.ndarray
  qvel : np.ndarray
  twist : Rotation
  translation : torch.Tensor 


  @classmethod
  def from_spec(cls, spec):
    jnt_type = JointType.from_mj(spec.type)

    if jnt_type == JointType.HINGE:
      return HingeJoint.from_spec(spec)
    if jnt_type == JointType.SLIDE:
      return SlideJoint.from_spec(spec)
    
    assert False

  @abstractmethod
  def transform(self, qpos = None):
    ...

  def __repr__(self):
    return f"<{__name__} {self.name} type={self.type} qpos={self.qpos.item():.2f} qvel={self.qvel.item():2f}]>"

@dataclass
class SlideJoint(Joint):
  axis : np.ndarray
  range : Tuple[float, float]
  type : JointType = JointType.SLIDE

  @classmethod
  def from_spec(cls, spec):
    return cls(
      name=spec.name,
      id=None,
      type=JointType.from_mj(spec.type),
      axis=torch.from_numpy(spec.axis.copy()),
      range=(spec.range[0], spec.range[1]),
      translation=torch.from_numpy(spec.pos.copy()),
      twist=Rotation(),
      qpos=np.zeros(1),
      qvel=np.zeros(1)
    )

  def transform(self, qpos = None):
    qpos = qpos if qpos is not None else self.qpos
    return Transform(
      position = self.translation + self.axis * qpos,
      rotation = self.twist
    )


@dataclass
class HingeJoint(Joint):
  axis : torch.Tensor
  range : Tuple[float, float]
  type : JointType = JointType.HINGE

  @classmethod
  def from_spec(cls, spec):
    return cls(
      name=spec.name,
      id=None,
      type=JointType.from_mj(spec.type),
      axis=torch.from_numpy(spec.axis.copy()),
      range=(spec.range[0], spec.range[1]),
      translation=torch.from_numpy(spec.pos.copy()),
      twist=Rotation.identity(),
      qpos=torch.zeros(1),
      qvel=torch.zeros(1)
    )
  
  def transform(self, qpos = None):
    qpos = qpos if qpos is not None else self.qpos
    return Transform(
      position = self.translation,
      rotation = self.twist * Rotation.from_rotvec(qpos * self.axis)
    )

