from tinysim.core import Robot, Scene, Simulation

from tinysim.web import WebRenderer

robot = Robot.load("panda")
scene = Scene.load("desk")

scene.attach(robot)


sim = Simulation(scene)


WebRenderer(sim, host="192.168.0.28")


while True:
  ...