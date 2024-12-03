import mujoco as mj

from tinysim.core.scene import Scene

class Simulation:

  def __init__(self, scene : Scene) -> None:
    
    self.scene = scene
    self.mj_model = scene.compile()
    self.mj_data = mj.MjData(self.mj_model)

    mj.mj_forward(self.mj_model, self.mj_data)