
from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


class   Rotation:
  def __init__(self, rotation : Union[np.ndarray, torch.Tensor] = None):
    
    if rotation is None:
      rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    if isinstance(rotation, np.ndarray):
      rotation = torch.from_numpy(rotation)

    assert isinstance(rotation, torch.Tensor)
    assert rotation.numel() == 4
    
    assert torch.norm(rotation) != 0

    self._data : torch.Tensor = rotation 

  def numpy(self) -> np.ndarray:
    return self._data.numpy()
  
  def rotate(self, vec : torch.Tensor) -> torch.Tensor:
    # Implement rotation using PyTorch operations
    q = self._data
    t = 2 * torch.linalg.cross(q[:3], vec.to(q.dtype))
    return vec + q[3] * t + torch.linalg.cross(q[:3], t)




  def as_quat(self) -> np.ndarray:
    return self._data.numpy()
  
  def inv(self):
    
    q_inv = self._data.clone()
    q_inv[:3] *= -1
    q_inv /= torch.norm(q_inv)

    return Rotation(q_inv)
  
  def copy(self):
    return Rotation(self._data.clone())

  def __mul__(self, other):
    # Quaternion multiplication using PyTorch operations
    q1 = self._data
    q2 = other._data
    w = q1[3] * q2[3] - torch.dot(q1[:3], q2[:3])
    xyz = q1[3] * q2[:3] + q2[3] * q1[:3] + torch.linalg.cross(q1[:3], q2[:3])
    return Rotation(torch.cat((xyz, w.unsqueeze(0))))


  def to_euler(self) -> torch.Tensor:
    # Convert quaternion to Euler angles using PyTorch operations
    q = self._data
    ysqr = q[1] * q[1]

    t0 = +2.0 * (q[3] * q[0] + q[1] * q[2])
    t1 = +1.0 - 2.0 * (q[0] * q[0] + ysqr)
    X = torch.atan2(t0, t1)

    t2 = +2.0 * (q[3] * q[1] - q[2] * q[0])
    t2 = torch.clamp(t2, -1.0, 1.0)
    Y = torch.asin(t2)

    t3 = +2.0 * (q[3] * q[2] + q[0] * q[1])
    t4 = +1.0 - 2.0 * (ysqr + q[2] * q[2])
    Z = torch.atan2(t3, t4)

    return torch.stack([X, Y, Z])
  

  def __repr__(self):
    return f"R[{self._data[0]:.4f}, {self._data[1]:.4f}, {self._data[2]:.4f}, {self._data[3]:.4f}]"
  
