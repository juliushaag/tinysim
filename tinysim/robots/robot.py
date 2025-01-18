from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import mujoco as mj
import numpy as np

from tinysim.scene.element import Element
from tinysim.simulation.body import SceneBody
import importlib
import inspect

from tinysim.core.transform import Rotation

import torch

ROBOTS_PATH = Path(__file__).parent
ROBOTS = { path.name : path / "robot.py" for path in ROBOTS_PATH.iterdir() if path.is_dir() and (path /  (path.name + ".py")).is_file()}


def load_robot(name : str) -> "Robot":
  if name not in ROBOTS:
    raise ValueError("Invalid robot, select one of", ROBOTS.keys())
  
  robot = importlib.import_module(f"tinysim.robots.{name}.{name}")
  members = inspect.getmembers(robot)
  
  robot_cls = next(cls for cname, cls in members if cname.lower().startswith(name.lower()) and inspect.isclass(cls) and issubclass(cls, Robot))

  return robot_cls()
 


class Robot(Element, ABC):

  ROBOTS = defaultdict(int)

  def __init__(self, kind : str, spec) -> None:

    name = f"{kind}:{Robot.ROBOTS[kind]}"
    Robot.ROBOTS[kind] += 1
    super().__init__(name, spec)

  def _on_simulation_init(self, sim):
    self._base_to_end_effector = list() 

    current = self.end_effector
    while current != self.base: 
      self._base_to_end_effector.append(current := current.parent)

    self._base_to_end_effector.reverse()
  

    super()._on_simulation_init(sim)


  def step(self):
    super().step()
    
  @property
  @abstractmethod
  def ctrl(self) -> np.ndarray:
    ...

  @ctrl.setter
  @abstractmethod
  def ctrl(self, ctrl : np.ndarray):
    ...

  
  @property
  @abstractmethod
  def base(self) -> SceneBody:
    ...

  @property
  @abstractmethod
  def end_effector(self) -> SceneBody:
    ...

  @property
  def chain(self) -> list[SceneBody]:
    return list(self._base_to_end_effector)

  def forward(self, qpos : np.ndarray = None, jacobian : np.ndarray = None) -> np.ndarray:

    if qpos is None: qpos = np.array([joint.qpos for joint in self.joints])

    assert len(qpos) == len(self.joints)
    assert jacobian is None or jacobian.shape == (7, len(self.joints))

    position = self.base.xpos
    rotation = self.base.xrot

    qpos = torch.from_numpy(qpos).requires_grad_(True)
    print(qpos)


    qpos_i = 0
    for body in self.chain:

      position += rotation.rotate(body.ipos)
      rotation *= body.irot

      # assert np.allclose(body.xpos.numpy(), position.numpy()), f"{body.name} {body.xpos.numpy()} != {position.numpy()} {body.irot}"
        
      for joint in body.joints:
        # position += rotation.rotate(joint.translation)
        # rotation *= joint.twist

        half_angle = qpos[qpos_i] / 2

        x, y, z = joint.axis * torch.sin(half_angle)
        rotation *= Rotation(torch.tensor([x, y, z, torch.cos(half_angle)]))
        qpos_i += 1


    return position + rotation.rotate(self.end_effector.ipos), rotation * self.end_effector.irot

  def pose2qpos(self, position : np.ndarray, step_length = 0.1) -> np.ndarray:

    # qpos = np.array([joint.qpos for joint in self.joints])

    # for i in range(1000):
    #   pos, rot = self.qpos2pose(qpos)
    #   while not np.allclose(pos, position, atol=1e-3):
    #     qpos -= step_length * 2 *  (position - self.qpos2pose(qpos)[0])


    # return qpos
    ...