import torch
import tinysim as ts

robot = ts.load_robot("panda")
env = ts.load_environment("desk")

env.attach(robot)

sim = ts.simulate(env)

for _ in range(1000):
  sim.step()

qpos = robot.inverse_kinematic([0.3, -0.4, 0.5], step_length=1)

sim.data.qpos = qpos.numpy()

while sim.is_running():
  sim.step()

  transform = robot.forward_kinematic()
  
  sim.data.qpos = qpos.numpy()
  sim.renderer.render_point(f"test", transform.position, size=torch.Tensor([0.1, 0.0, 0.0]))