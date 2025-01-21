import tinysim as ts

robot = ts.load_robot("panda")
env = ts.load_environment("desk")

env.attach(robot)

sim = ts.simulate(env, renderer=None)

for _ in range(1000):
  sim.step()

qpos = robot.inverse_kinematic([0.5, 0.5, 0.5], step_length=0.002)

robot.ctrl = qpos[:8]


while sim.is_running():
  sim.step()
  print(robot.end_effector.xpos)
