import numpy as np
import torch
from tinysim.core.transform import Rotation, Transform
from scipy.spatial.transform import Rotation as R

def tt(any) -> torch.Tensor:
  if isinstance(any, np.ndarray):
    return torch.from_numpy(any)
  if isinstance(any, R):
    return torch.from_numpy(any.as_quat())
  
  assert False 

def test_rotation():
  for _ in range(100):
    quat1 = R.from_euler("xyz", np.random.rand(3))
    quat2 = R.from_euler("xyz", np.random.rand(3))

    assert np.allclose((quat2 * quat1).as_quat(), (Rotation(tt(quat2)) * Rotation(tt(quat1))).to_quat())

    assert np.allclose((quat1.inv() * quat2).as_quat(), (Rotation(tt(quat1)).inv() * Rotation(tt(quat2))).to_quat())

    assert np.allclose(quat1.inv().as_quat(), Rotation(tt(quat1)).inv().to_quat())


    assert np.allclose(quat1.as_euler("xyz"), Rotation(tt(quat1)).to_euler())

    assert np.allclose(quat1.as_quat(), Rotation.from_rotvec(tt(quat1.as_rotvec())).to_quat())


    pos = torch.rand(3) * 100

    assert np.allclose(quat1.apply(pos.numpy()), Rotation(tt(quat1)).apply(pos).numpy())

def test_transform():

  for _ in range(100):
    quat1 = R.from_euler("xyz", np.random.rand(3))

    pos = torch.rand(3) * 100
    pos1 = torch.rand(3) * 100

    assert np.allclose(quat1.apply(pos.numpy()) + pos1.numpy(), Transform(pos1, Rotation(tt(quat1))).apply(pos))

if __name__ == "__main__":
  test_rotation()