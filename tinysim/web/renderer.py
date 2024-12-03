from dataclasses import asdict, is_dataclass
import io
import json
from pathlib import Path
from threading import Thread
import flask
import logging

import numpy as np

from tinysim.core import Simulation

WEB_PATH = Path(__file__).parent.resolve()

class JsonEncoder(json.JSONEncoder):
  def default(self, obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.generic):
        return obj.item()
      elif is_dataclass(obj):
        return asdict(obj)
      return super().default(obj)

class WebRenderer:

  def __init__(self, simulation : Simulation, host="127.0.0.1", port = 5000):
    
    self.host = host
    self.port = port

    self.sim = simulation

    self.app = flask.Flask("TinySim", template_folder=WEB_PATH / "template", static_folder=WEB_PATH / "static")
    
    # log = logging.getLogger('werkzeug')
    # log.setLevel(logging.ERROR)

    # log = logging.getLogger('flask')
    # log.setLevel(logging.ERROR)
    
    self.app.add_url_rule('/', 'index', self.get_index)
    self.app.add_url_rule('/scene_id', 'scene_id', self.get_scene_id)
    self.app.add_url_rule('/scene_data', 'scene_data', self.get_scene)
    self.app.add_url_rule('/scene_state', 'scene_state', self.get_state)
    self.app.add_url_rule('/data/<hash>', 'data', self.get_data)

    self.scene_info = json.dumps(dict(
      root = self.sim.scene.root,
      id = self.sim.scene.id,
      meshes = self.sim.scene.meshes,
      materials = self.sim.scene.materials,
      textures = self.sim.scene.textures
    ), cls=JsonEncoder)

    self.scene_id = self.sim.scene.id
  
    Thread(target=self.app.run, kwargs=dict(host=host, port=port, debug=False)).start()


  """
  Webserver connection
  """
  def get_index(self):
    return flask.render_template("index.html")

  def get_scene_id(self): 
    return str(self.scene_id)
  
  def get_scene(self):
    return self.scene_info or {}
  
  def get_state(self):
    return {}
  
  def get_data(self, hash):
    data = self.sim.scene.assets[hash]
    if data is not None:
      return flask.send_file(io.BytesIO(data), mimetype='blob/bin')
    return "Invalid asset data request", 404
  