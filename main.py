import tinysim as ts
import mujoco
import mediapy as media
ts.set_seed(0)


panda_rb = ts.load_robot("panda")
scene = ts.load_scene("desk")

scene.attach(panda_rb)

scene.compile()
model = scene.mj_model
data = mujoco.MjData(model)