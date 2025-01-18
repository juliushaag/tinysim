
from dataclasses import dataclass
from hashlib import md5

import numpy as np

from tinysim.simulation.body import SceneBody


class Element:
  def __init__(self, name : str, spec) -> None:
    self._name = name
    self._spec = spec

    self._root = SceneBody.from_spec(spec.worldbody)

    self._attached_elements = list()
  
  @property
  def name(self):
    return self._name
  
  @property
  def root(self):
    return self._root
  
  @property
  def bodies(self):
    return self._root.get_all_bodies()
  
  @property
  def joints(self):
    return self._root.get_all_joints()
  
  def body(self, ident : int | str) -> SceneBody:
    return next(body for body in self.bodies if body.name == ident or body.id == ident or body.name == f"{self.name}{ident}")
  
  def joint(self, ident : int | str) -> SceneBody:
    return next(joint for joint in self.joints if joint.name == ident or joint.id == ident or joint.name == f"{self.name}{ident}")

  def attach(self, element : "Element", mount_point : SceneBody):

    assert mount_point in self.bodies
    mount_point.attach(element._root, element.name)

    # body names are unique, so we need to rename them (2 pandas in one scene etc)
    for body in element.bodies:
      body.name = f"{element.name}{body.name}"

    # same with joints
    for joint in element.joints:
      joint.name = f"{element.name}{joint.name}"

    self._attached_elements.append(element)  

  def compile(self):

    model = self._spec.compile() 
    self.compiled = True
    self.id: str = md5(self.env_spec.to_xml().encode()).hexdigest()


    for body in self.bodies:
      model_body = model.body(body.name)
      body.id = model_body.id
      body.position_rel = model_body.pos
      body.rotation_rel = model_body.quat

    for joint in self.joints:
      joint.id = model.jnt(joint.name).id

    return model
  

  def _on_simulation_init(self, sim):
    for element in self._attached_elements:
      element._on_simulation_init(sim)

  def step(self):
    for element in self._attached_elements:
      element.step()