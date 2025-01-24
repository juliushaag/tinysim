
import torch



@torch.jit.script
class Rotation:
  def __init__(self, rotation : torch.Tensor = None):
    
    if rotation is None:
      rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    assert isinstance(rotation, torch.Tensor)
    assert rotation.numel() == 4


    self._data : torch.Tensor = rotation / torch.norm(rotation)

  @classmethod
  def identity(cls) -> "Rotation":
    return cls(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
  
  @classmethod
  def from_rotvec(cls, rotvec: torch.Tensor) -> "Rotation":
    angle = torch.norm(rotvec)
    if angle == 0:
      return cls(torch.cat([rotvec, torch.tensor([1.0])]))
    axis = rotvec / angle
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


  def to_matrix(self) -> torch.Tensor:
    q = self._data
    q0, q1, q2, q3 = q[3], q[0], q[1], q[2]
    R = torch.stack([
      1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
      2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1),
      2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)
    ]).reshape(3, 3)

    return R

  @classmethod
  def from_matrix(cls, matrix: torch.Tensor) -> "Rotation":
    assert matrix.shape == (3, 3)
    R = matrix
    trace = R.trace()
    if trace > 0:
      s = 2.0 * torch.sqrt(trace + 1.0)
      qw = 0.25 * s
      qx = (R[2, 1] - R[1, 2]) / s
      qy = (R[0, 2] - R[2, 0]) / s
      qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
      s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
      qw = (R[2, 1] - R[1, 2]) / s
      qx = 0.25 * s
      qy = (R[0, 1] + R[1, 0]) / s
      qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
      s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
      qw = (R[0, 2] - R[2, 0]) / s
      qx = (R[0, 1] + R[1, 0]) / s
      qy = 0.25 * s
      qz = (R[1, 2] + R[2, 1]) / s
    else:
      s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
      qw = (R[1, 0] - R[0, 1]) / s
      qx = (R[0, 2] + R[2, 0]) / s
      qy = (R[1, 2] + R[2, 1]) / s
      qz = 0.25 * s

    quat = torch.tensor([qx, qy, qz, qw], dtype=torch.float64)
    return cls(quat)

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
    return self._position + self._rotation.apply(vec)
  
  def __mul__(self, other: "Transform") -> "Transform":
    return Transform(
      position=self.apply(other._position),
      rotation=self._rotation * other._rotation,
    )

  def inv(self) -> "Transform":
    inv = self._rotation.inv()
    return Transform(rotation=inv, position=-inv.apply(self._position))
  
  def copy(self) -> "Transform":
    return Transform(position=self.position, rotation=self.rotation)
  
  @property
  def position(self):
    return self._position.clone()
  
  @property
  def rotation(self):
    return self._rotation.copy()
  
