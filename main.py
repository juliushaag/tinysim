import tinysim as ts
import mujoco
ts.set_seed(0)


panda_rb = ts.load_robot("panda")
scene = ts.load_environment("desk")

scene.attach(panda_rb)

scene.compile()