from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import mujoco as mj
import numpy as np

from tinysim.scene.element import Element
from tinysim.simulation.body import SceneBody
import importlib
import inspect

from tinysim.core.transform import Rotation, Transform, chain_transforms

import torch
import matplotlib.pyplot as plt
from torchviz import make_dot

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

  def forward_kinematic(self, qpos : torch.Tensor = None) -> Transform:

    qpos : torch.Tensor = qpos if qpos is not None else torch.tensor([joint.qpos.item() for joint in self.joints])

    assert len(qpos) == len(self.joints)
    transform = self.base.xtransform.copy()


    transform = [self.base.xtransform]
    for body in self.chain:
      transform.append(body.itransform)
      for joint in body.joints:
        transform.append(joint.transform(qpos[joint.id]))

    transform.append(self.end_effector.itransform)
    
    return chain_transforms(transform)

  def inverse_kinematic(self, position : list, step_length = 0.01) -> torch.Tensor:

    qpos = torch.tensor([joint.qpos.item() for joint in self.joints]).requires_grad_()
    
    position = torch.tensor(position, dtype=torch.float64) 

    jacobian = torch.zeros((3, len(self.joints)), dtype=torch.float64)

    transform = self.forward_kinematic(qpos)

    for i in range(3):
      grad  = torch.autograd.grad(transform.position[i], [qpos], create_graph=True)[0]
      jacobian[i] = grad

    while not np.allclose(transform.position.detach().numpy(), position, atol=1e-4):

      loss = (transform.position - position)
      qpos = qpos - step_length * 2 * (jacobian.T @ loss)
      
      transform = self.forward_kinematic(qpos)
      for i in range(3):
        grad  = torch.autograd.grad(transform.position[i], [qpos], create_graph=True)[0]
        jacobian[i] = grad
 
    return qpos.detach()
    