from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
import mujoco as mj
import torch

from tinysim.core.transform import Position, Rotation

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
class Joint:
  id : int
  name : str
  qpos : np.ndarray
  qvel : np.ndarray
  twist : Rotation
  translation : Position 


  @classmethod
  def from_spec(cls, spec):
    jnt_type = JointType.from_mj(spec.type)

    if jnt_type == JointType.HINGE:
      return HingeJoint.from_spec(spec)
    if jnt_type == JointType.SLIDE:
      return SlideJoint.from_spec(spec)
    
    assert False

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
      axis=Position(spec.axis.copy()),
      range=(spec.range[0], spec.range[1]),
      translation=Position(spec.pos.copy()),
      twist=Rotation(),
      qpos=np.zeros(1),
      qvel=np.zeros(1)
    )

@dataclass
class HingeJoint(Joint):
  axis : Position
  range : Tuple[float, float]
  type : JointType = JointType.HINGE

  @classmethod
  def from_spec(cls, spec):
    return cls(
      name=spec.name,
      id=None,
      type=JointType.from_mj(spec.type),
      axis=Position(spec.axis.copy()),
      range=(spec.range[0], spec.range[1]),
      translation=Position(spec.pos.copy()),
      twist=Rotation(),
      qpos=np.zeros(1),
      qvel=np.zeros(1)
    )