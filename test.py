import time
import tinysim as ts


robot = ts.load_robot("panda")
scene = ts.load_scene("desk")

scene.attach(robot)


sim = ts.simulate(scene, render_args=dict(host="192.168.0.28"))

while True:
  sim.step()
