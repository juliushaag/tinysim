from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
import mujoco as mj

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
      id =None,
      type=SceneJointType.from_mj(spec.type),
      pos=spec.pos,
      axis=spec.axis,
      range=(spec.range[0], spec.range[1])
    )

  def __repr__(self):
    return f"<SceneJoint {self.name} type={self.type} range=[{self.range[0].item()}:{self.range[1].item()}]>"
