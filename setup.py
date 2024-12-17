#!/usr/bin/env python

from setuptools import setup

setup(
  name='tinysim',
  version='0.1',
  description="Implementation of robotic basics using mujoco",
  author='Julius Haag',
  author_email='haag.julius@outlook.de',
  packages=['tinysim.models'],
  requires=['mujoco', "trimesh", "pyyaml", "websockets"]
)