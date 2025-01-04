from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import mujoco as mj
import numpy as np
import yaml

from tinysim.scene.element import Element

ROBOTS_PATH = Path(__file__).parent / "../../models/robots" 
ROBOTS = { path.name : path for path in ROBOTS_PATH.iterdir() }


@dataclass
class RobotConfig:
  definition: str
  torque_limits: list[float]
  joint_vel_limit: list[float]
  joint_pos_min: list[float]
  joint_pos_max: list[float]

def load_robot(name : str) -> "Robot":
  if name not in ROBOTS:
    raise ValueError("Invalid robot, select one of", ROBOTS.keys())
  
  conf = ROBOTS[name] / "description.yaml"

  if not conf.is_file():
    raise ValueError("No 'description.yaml' found for", name)

  robot_config = RobotConfig(**yaml.full_load(conf.read_text()))
  robot_config.definition = str(ROBOTS[name] / robot_config.definition)

  kind = name
  name = f"{name}:{Robot.ROBOTS[name]}"
  Robot.ROBOTS[name] += 1

  return Robot(name, kind, robot_config)

class Robot(Element):

  ROBOTS = defaultdict(int)

  def __init__(self, name : str, kind, conf : RobotConfig) -> None:
    
    spec = mj.MjSpec.from_file(conf.definition)
    super().__init__(name, spec)

    self._conf = conf    
    self._ctrl = None

    self.kind = kind

    self.ee = self.bodies[-1]
    self.base = self.bodies[0]
  
  def _on_simulation_init(self, sim):
    model = sim.model

    jnt_ids = set(joint.id for joint in self.joints)
    self.acts_idx = [model.actuator(i).id for i in range(model.nu) if model.actuator(i).trnid[0] in jnt_ids]
    self._ctrl = np.zeros(len(self.acts_idx))

    self.sim = sim
    super()._on_simulation_init(sim)

  def step(self):
    self.sim.data.ctrl[self.acts_idx] = self._ctrl
    super().step()
    
  @property
  def ctrl(self):
    return self._ctrl.copy()
  
  @ctrl.setter
  def ctrl(self, ctrl : np.ndarray):
    assert len(ctrl) == len(self._ctrl), "Control vector must have the same length as the number of joints"
    self._ctrl[:] = ctrl

  def get_ee_pose(self):
    return self.ee.position, self.ee.quaternion
  
  def get_base_pos(self):
    return self.base.position, self.base.quaternion
  