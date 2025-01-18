from pathlib import Path

import numpy as np
from tinysim.robots.robot import Robot
import mujoco

from tinysim.simulation.body import SceneBody

class PandaRobot(Robot):

  def __init__(self):

    self._spec_file = Path(__file__).parent / "panda.xml"
    self._spec = mujoco.MjSpec.from_file(self._spec_file.__str__())

    super().__init__("panda", self._spec)

  def _on_simulation_init(self, sim):

    model = sim.model

    self._acts_idx = [model.actuator(self.name + "actuator" + str(i)).id for i in range(1, 9)]
    self._ctrl = np.zeros(len(self._acts_idx))

    self._ee_body : SceneBody = self.body("hand")
    self._base_body : SceneBody = self.body("link0")

    self._ctrl[:] = model.key(self.name+"home").ctrl

    self.sim = sim
    super()._on_simulation_init(sim)  

  def step(self):
    self.sim.data.ctrl[self._acts_idx] = self._ctrl
    return super().step()
  
  @property
  def ctrl(self) -> np.ndarray:
    return self._ctrl
  
  @ctrl.setter
  def ctrl(self, ctrl : np.ndarray):
    self._ctrl = ctrl
    
  @property
  def base(self) -> SceneBody:
    return self._base_body

  @property
  def end_effector(self) -> SceneBody:
    return self._ee_body