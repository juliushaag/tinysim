import tinysim as ts

robot = ts.load_robot("kuka")
env = ts.load_environment("plane")

env.attach(robot)


sim = ts.simulate(env)

# start_pos = sim.model.key(0).ctrl

# robot.ctrl = start_pos

def interpolate(start, end, t):
  return start + (end - start) * t



while sim.is_running():
  sim.step()
  print(robot.get_ee_pose())