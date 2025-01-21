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

from tinysim.core.transform import Rotation

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

  def forward_kinematic(self, qpos : torch.tensor = None, jacobian : np.ndarray = None) -> Tuple[torch.Tensor, Rotation]:

    if qpos is None: qpos = torch.tensor([joint.qpos.item() for joint in self.joints])

    assert len(qpos) == len(self.joints)
    assert jacobian is None or jacobian.shape == (6, len(self.joints))

    position = self.base.xpos
    rotation = self.base.xrot

    qpos : torch.Tensor = qpos.clone().requires_grad_()

    qpos_i = 0
    for body in self.chain:

      position = position + rotation.rotate(body.ipos)
      rotation = rotation * body.irot

   

        
      for joint in body.joints:
        position = position + rotation.rotate(joint.translation)
        rotation = rotation *  joint.twist

        half_angle = qpos[qpos_i] / 2
        quat = torch.cat([half_angle.sin() * joint.axis, half_angle.cos().unsqueeze(0)])
        rotation = rotation * Rotation(quat)

        qpos_i += 1


    position = position + rotation.rotate(self.end_effector.ipos)
    rotation = rotation * self.end_effector.irot

    euler = rotation.to_euler()

    if jacobian is not None:
      for i in range(3):
        grad  = torch.autograd.grad(position[i], [qpos], create_graph=True)[0]
        jacobian[i] = grad

      for i in range(3):
        grad = torch.autograd.grad(euler[i], [qpos], create_graph=True)[0]
        jacobian[i + 3] = grad

    return position, rotation

  def inverse_kinematic(self, position : list, step_length = 0.01) -> torch.Tensor:

    qpos = torch.tensor([joint.qpos.item() for joint in self.joints])
    
    position = torch.tensor(position, dtype=torch.float64) 

    jacobian = torch.zeros((6, len(self.joints)), dtype=torch.float64)

    pos, rot = self.forward_kinematic(qpos=qpos, jacobian=jacobian)
    while not np.allclose(pos.detach().numpy(), position, atol=1e-5):

      loss = (pos - position)
      qpos = qpos - step_length * 2 * (jacobian[:3].T @ loss)
      
      pos, rot = self.forward_kinematic(qpos=qpos, jacobian=jacobian)
 
    return qpos.detach()
    