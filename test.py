import tinysim as ts

robot = ts.load_robot("panda")
env = ts.load_environment("desk")

env.attach(robot)

sim = ts.simulate(env)


while sim.is_running():
  sim.step()

  position, rotation = robot.forward()

  sim.get_renderer().render_point(f"test{0}", position)


