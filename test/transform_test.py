import numpy as np
import torch
from tinysim.core.transform import Position, Rotation
from scipy.spatial.transform import Rotation as R

def test_transform():
  for _ in range(100):
    quat1 = R.from_euler("xyz", np.random.rand(3))
    quat2 = R.from_euler("xyz", np.random.rand(3))

    assert np.allclose((quat1 * quat2).as_quat(), (Rotation(quat1.as_quat()) * Rotation(quat2.as_quat())).as_quat())
    
    assert np.allclose((quat1.inv() * quat2).as_quat(), (Rotation(quat1.as_quat()).inv() * Rotation(quat2.as_quat())).as_quat())

    assert np.allclose(quat1.inv().as_quat(), Rotation(quat1.inv().as_quat()).as_quat())

    pos = Position(np.random.rand(3) * 100)
    pos1 = Position(np.random.rand(3) * 100)

    assert np.allclose(quat1.apply(pos.numpy()), Rotation(quat1.as_quat()).rotate(pos).numpy())

    assert np.allclose(pos.numpy() + pos1.numpy(), (pos + pos1).numpy())

    assert np.allclose(pos.numpy() - pos1.numpy(), (pos - pos1).numpy())

    assert np.allclose(pos.numpy() * pos1.numpy(), (pos * pos1).numpy())


    random_v = torch.randn(1)
    assert np.allclose(random_v * pos1.numpy(), (random_v * pos1).numpy())

if __name__ == "__main__":
  test_transform()