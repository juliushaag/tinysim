
from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

class Position:
  def __init__(self, position : np.ndarray | torch.Tensor = None):
    if position is None:
      position = torch.zeros(3)
    if isinstance(position, np.ndarray):
      position = torch.from_numpy(position)
    assert isinstance(position, torch.Tensor) 
    assert position.shape == (3,)
    self._data : torch.Tensor = position

  def numpy(self) -> np.ndarray:
    return self._data.numpy()
  
  def __add__(self, other):
    assert isinstance(other, Position)
    return Position(self._data + other._data)
  
  def __sub__(self, other):
    return self + (-other)
  
  def __neg__(self):
    return Position(-self._data)
  
  def __mul__(self, other) -> "Position": 
    if isinstance(other, torch.Tensor | np.ndarray | float):
      return Position(self._data * torch.tensor(other))
     
    return Position(self._data * other._data)
  
  def __rmul__(self, other):
    return self * other
  
  def __iter__(self):
    return iter(self._data)
  
  def __repr__(self):
    return f"P[{self._data[0]:.4f}, {self._data[1]:.4f}, {self._data[2]:.4f}]"

class   Rotation:
  def __init__(self, rotation : Union[np.ndarray, torch.Tensor] = None):
    
    if rotation is None:
      rotation = torch.tensor([0.0, 0.0, 0.0, 1.0])
    if isinstance(rotation, np.ndarray):
      rotation = torch.from_numpy(rotation)

    assert isinstance(rotation, torch.Tensor)
    assert rotation.shape == (4,)
    
    self._data : torch.Tensor = rotation
  

  def numpy(self) -> np.ndarray:
    return self._data.numpy()

  @classmethod
  def from_matrix(cls, matrix : np.ndarray | torch.Tensor):
    if isinstance(matrix, np.ndarray):
      matrix = torch.from_numpy(matrix)
    assert isinstance(matrix, torch.Tensor)
    assert matrix.shape == (3, 3)


    w = torch.sqrt(1.0 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2.0

    w4 = (4.0 * w)

    x = (matrix[2, 1] - matrix[1, 2]) / w4
    y = (matrix[0, 2] - matrix[2, 0]) / w4
    z = (matrix[1, 0] - matrix[0, 1]) / w4
      
    return cls(torch.tensor([x, y, z, w]))
  
  def rotate(self, vector : Position) -> Position:
    assert isinstance(vector, Position)

    return Position((self * Rotation(torch.concat([vector._data, torch.tensor([0.0])])) * self.inv())._data[:3])

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
    assert type(other) == type(self)

    r = other._data[[3, 0, 1, 2]]
    q = self._data[[3, 0, 1, 2]]

    return Rotation(
      torch.tensor([
       r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2], # x
       r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1], # y
       r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0], # z

       r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3], # w
      ])
    )
  

  def __repr__(self):
    return f"R[{self._data[0]:.4f}, {self._data[1]:.4f}, {self._data[2]:.4f}, {self._data[3]:.4f}]"
  
