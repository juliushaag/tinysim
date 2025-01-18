
from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


class   Rotation:
  def __init__(self, rotation : Union[np.ndarray, torch.Tensor] = None):
    
    if rotation is None:
      rotation = torch.tensor([0.0, 0.0, 0.0, 1.0])
    if isinstance(rotation, np.ndarray):
      rotation = torch.from_numpy(rotation)

    assert isinstance(rotation, torch.Tensor)
    assert rotation.numel() == 4
    
    assert torch.norm(rotation) != 0

    self._data : torch.Tensor = rotation 

  def numpy(self) -> np.ndarray:
    return self._data.numpy()
  
  def rotate(self, vector : torch.Tensor) -> torch.Tensor:
    assert isinstance(vector, torch.Tensor)
    # return (self * Rotation(, normalize=False) * self.inv())._data[:3]
  
    r = torch.tensor([0.0, *vector])
    q = self._data[[3, 0, 1, 2]]

    q = torch.tensor([
      r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3], # w
    
      r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2], # x
      r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1], # y
      r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0], # z
    ])

    r = self.inv()._data[[3, 0, 1, 2]]

    return torch.tensor([
      r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2], # x
      r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1], # y
      r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0], # z
    ])



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
    assert isinstance(other, Rotation)

    r = other._data[[3, 0, 1, 2]]
    q = self._data[[3, 0, 1, 2]]

    quat = torch.tensor([
      r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2], # x
      r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1], # y
      r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0], # z

      r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3], # w
    ])
     
    quat /= torch.norm(quat)

    return Rotation(quat)
  

  def __repr__(self):
    return f"R[{self._data[0]:.4f}, {self._data[1]:.4f}, {self._data[2]:.4f}, {self._data[3]:.4f}]"
  
