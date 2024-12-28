from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import mujoco as mj
import yaml

from tinysim.scene.scene import SceneElement

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

  robot_name = f"{name}:{ROBOTS[name]}"
  Robot.ROBOTS[name] += 1

  return Robot(name, robot_config)

class Robot(SceneElement):
  ROBOTS = defaultdict(int)

  def __init__(self, name : str, conf : RobotConfig) -> None:
    spec = mj.MjSpec.from_file(conf.definition)
    super().__init__(name, spec)
    self._conf = conf    
    self._ctrl = None

  def get_body_by_name(self, name):
    if name not in self._bodies:
      name = f"{self.name}{name}" # prepend namespace
    return super().get_body_by_name(name)
  
  def _on_simulation_init(self):
    self.joint_act_ind = [joint.id for joint in self.joints]

  def get_ctrl(self, reset = False):
    ctrl = self._ctrl
    self._ctrl = None if reset else ctrl
    return ctrl
  
  def set_ctrl(self, ctrl):
    assert len(ctrl) == len(self.joints), "Control vector must have the same length as the number of joints"
    self._ctrl = ctrl
    