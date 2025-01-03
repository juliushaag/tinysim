from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Optional, Union
import mujoco as mj
import yaml
from tinysim.scene.robot import Robot
from tinysim.scene.scene import SceneBody, SceneElement


ENVIRONMENTS_PATH = (Path(__file__).parent / "../../models/environments").resolve()
ENVIRONMENT = { path.name : path for path in ENVIRONMENTS_PATH.iterdir() }

@dataclass
class EnvironmentConfig:
  definition: str
  robot_mount_points: Union[str, list]

def load_environment(name : str) -> "Environment":
  
  if name not in ENVIRONMENT:
    raise ValueError("Invalid scene, select one of", ENVIRONMENT.keys())
  
  conf = ENVIRONMENT[name] / "description.yaml"

  if not conf.is_file():
    raise ValueError("No 'description.yaml' found for", name)

  scene_config = EnvironmentConfig(**yaml.full_load(conf.read_text()))



  scene_config.definition = str(ENVIRONMENT[name] / scene_config.definition)
  env_spec = mj.MjSpec.from_file(str(scene_config.definition))

  return Environment(name, env_spec, scene_config)

def load_xml(xml : str) -> "Environment":
  return Environment.from_xml(xml)


class Environment(SceneElement):
  
  
  def __init__(self, name : str, spec, conf : EnvironmentConfig) -> None:
    self.conf = conf
    self.env_spec = spec
    super().__init__(name, self.env_spec)

    mount_points = self.conf.robot_mount_points if isinstance(self.conf.robot_mount_points, list) else [self.conf.robot_mount_points]

    self.robots : list[Robot] = []
    self.mount_points : dict[str, Optional[Robot]]= { name : None for name in mount_points }

  @classmethod
  def from_xml(self, xml : str):
    return Environment("custom", mj.MjSpec.from_string(xml), EnvironmentConfig("custom.xml", "robot"))
  
  def attach(self, robot : Robot, mount_point = None):

    if mount_point:
      assert self.mount_points[mount_point] == None, f"Mount point {mount_point} is already occupied"
      self.mount_points[mount_point] = robot
    else:
      free_mounts = [key for key in self.mount_points.keys() if self.mount_points[key] is None]
      assert len(free_mounts) > 0, "Not more free mounts to attach to"
      mount_point = free_mounts[0]
      self.mount_points[mount_point] = robot


    mp = self._bodies.get(mount_point)
    robot._reload(mp.attach(robot._root, robot.name))
    self._reload(SceneBody.from_spec(self._root.spec))

    self.robots.append(robot)

  def _on_simulation_init(self):
    for robot in self.robots:
      robot._on_simulation_init()

  def _reset_from_model(self, model):
    self._reload(SceneBody.from_model(model))

    for robot in self.robots:
        robot._reload(self._bodies.get(robot.root.name))

  def _compile(self):
    mj_model = self.env_spec.compile()
    self.compiled = True
    self.id: str = md5(self.env_spec.to_xml().encode()).hexdigest()
    self._reset_from_model(mj_model)
    return mj_model
    
  def get_body_by_name(self, name : str) -> SceneBody:
    return self.name_to_body[name]
