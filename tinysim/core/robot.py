from dataclasses import dataclass
from pathlib import Path
import mujoco as mj
import yaml
ROBOTS_PATH = Path(__file__).parent / "../../models/robots"
 
ROBOTS = { path.name : path for path in ROBOTS_PATH.iterdir() }


@dataclass
class RobotConfig:
  definition: str
  torque_limits: list[float]
  joint_vel_limit: list[float]
  joint_pos_min: list[float]
  joint_pos_max: list[float]

class Robot():

  @classmethod
  def load(cls, robot_name : str) -> "Robot":
    
    if robot_name not in ROBOTS:
      raise ValueError("Invalid robot, select one of", ROBOTS.keys())
    
    conf = ROBOTS[robot_name] / "description.yaml"

    if not conf.is_file():
      raise ValueError("No 'description.yaml' found for", robot_name)

    robot_config = RobotConfig(**yaml.full_load(conf.read_text()))
    robot_config.definition = str(ROBOTS[robot_name] / robot_config.definition)


    return cls(robot_name, robot_config)


  def __init__(self, name : str, conf : RobotConfig) -> None:

    self.name = name

    self.conf = conf    
    self.xml_file = Path(conf.definition)

  def get_xml_file(self) -> str:
    return self.xml_file

  def get_spec(self) -> mj.MjSpec:
    return mj.MjSpec.from_file(self.xml_file)
