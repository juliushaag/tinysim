import math
import time

import numpy as np
import tinysim as ts


robot = ts.load_robot("panda")
env = ts.load_environment("desk")

env.attach(robot)


sim = ts.simulate(env, render_args=dict(host="192.168.0.28"))

sim.load_environment(env)

start = time.monotonic()
sections = 1000 * 10 # 100s
interval = sections / len(sim.model.key(0).ctrl)

start_pos = sim.model.key(0).ctrl


while True:
  # sim.data.ctrl[:] = start_pos
  sim.step()
  # t = (time.monotonic() - start) % sections

  # idx = int(t / interval)

  # if t > 10: exit()
  # start_pos[idx] = math.sin((t / interval) * math.pi * 2) * 0.1