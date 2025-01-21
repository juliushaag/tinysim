
from typing import Optional, Self, Union
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch



@torch.jit.script
class Rotation:
  def __init__(self, rotation : torch.Tensor = None):
    
    if rotation is None:
      rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    assert isinstance(rotation, torch.Tensor)
    assert rotation.numel() == 4
    
    assert torch.norm(rotation) != 0

    self._data : torch.Tensor = rotation 
  
  @classmethod
  def from_angle_axis(cls, angle : torch.Tensor, axis : torch.Tensor) -> "Rotation":
    half_angle = angle / 2
    return cls(torch.cat([half_angle.sin() * axis, half_angle.cos().unsqueeze(0)]))
    
  
  def apply(self, vec : torch.Tensor) -> torch.Tensor:
    assert vec.numel() == 3
    q = self._data
    t = 2 * torch.linalg.cross(q[:3], vec.to(q.dtype))
    return vec + q[3] * t + torch.linalg.cross(q[:3], t)
  
  def inv(self):
    
    q_inv = self._data.clone()
    q_inv[:3] *= -1
    q_inv /= torch.norm(q_inv)

    return Rotation(q_inv)
  
  def copy(self):
    return Rotation(self._data.clone())

  def __mul__(self, other : "Rotation") -> "Rotation":
    # Quaternion multiplication using PyTorch operations
    q1 = self._data
    q2 = other._data
    w = q1[3] * q2[3] - torch.dot(q1[:3], q2[:3])
    xyz = q1[3] * q2[:3] + q2[3] * q1[:3] + torch.linalg.cross(q1[:3], q2[:3])
    return Rotation(torch.cat((xyz, w.unsqueeze(0))))

  def to_quat(self) -> torch.Tensor:
    return self._data.clone()

  def to_euler(self) -> torch.Tensor:
    # Convert quaternion to Euler angles using PyTorch operations
    q = self._data
    ysqr = q[1] * q[1]

    t0 = 2.0 * (q[3] * q[0] + q[1] * q[2])
    t1 = 1.0 - 2.0 * (q[0] * q[0] + ysqr)
    X = torch.atan2(t0, t1)

    t2 = 2.0 * (q[3] * q[1] - q[2] * q[0])
    t2 = torch.clamp(t2, -1.0, 1.0)
    Y = torch.asin(t2)

    t3 = 2.0 * (q[3] * q[2] + q[0] * q[1])
    t4 = 1.0 - 2.0 * (ysqr + q[2] * q[2])
    Z = torch.atan2(t3, t4)

    return torch.stack([X, Y, Z])


@torch.jit.script
class Transform:
  def __init__(self, position : torch.Tensor, rotation : Rotation):
    self._position = position
    self._rotation = rotation
  
  @classmethod
  def idenity(cls) -> "Transform":
    return cls(
      position=torch.zeros(3, dtype=torch.float64),
      rotation=Rotation(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    )
  def apply(self, vec : torch.Tensor) -> torch.Tensor:
    return self._rotation.apply(vec) + self._position
  
  def __mul__(self, other: "Transform") -> "Transform":
    return Transform(
      position=self._rotation.apply(other._position) + self._position,
      rotation=self._rotation * other._rotation,
    )

  def inv(self) -> "Transform":
    inv = self._rotation.inv()
    return Transform(rotation=inv, position=-inv.apply(self._position))
  
  def copy(self) -> "Transform":
    return Transform(position=self._position.clone(), rotation=self._rotation.copy())
  
  @property
  def position(self):
    return self._position.clone()
  
  @property
  def rotation(self):
    return self._rotation.copy()
  
@torch.jit.script
def chain_transforms(transforms : list[Transform]) -> Transform:
  result = transforms[0]
  for t in transforms[1:]:
    result = result * t
  return result