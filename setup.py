#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
  name='tinysim',
  version='0.1',
  description="Implementation of robotic basics using mujoco",
  author='Julius Haag',
  author_email='haag.julius@outlook.de',
  packages=find_packages(),
  requires=['mujoco', "trimesh", "pyyaml", "websockets"]
)