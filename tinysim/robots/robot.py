from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import mujoco as mj
import numpy as np
import yaml

from tinysim.scene.element import Element
import importlib
import inspect

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

  def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
    ...

  def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
    ...
  