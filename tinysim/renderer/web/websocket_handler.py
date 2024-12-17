from abc import ABC
from dataclasses import asdict, is_dataclass
import json
import time
import numpy as np
from websockets.sync.server import serve
from websockets.sync.server import ServerConnection

class JsonEncoder(json.JSONEncoder):
  def default(self, obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.generic):
        return obj.item()
      elif is_dataclass(obj):
        return asdict(obj)
      return super().default(obj)


class WebSocketConnection(ABC):

  def __init__(self, ws):
    self.ws : ServerConnection = ws
    self.connected = True

  def send(self, topic, message = ""):
    try:
      if not isinstance(message, str):
        message = json.dumps(message, cls=JsonEncoder)
      self.ws.send(topic + ":" + message)
    except Exception as e: 
      self.connected = False

  def recv(self):
    return self.ws.recv()

class WebSocketServer:

  def __init__(self, host : str, port : int, on_connection) -> None:
    self.host = host
    self.port = port

    self.on_connection = on_connection or (lambda x: None)

  def _on_connection(self, websocket):
    conn = WebSocketConnection(websocket)
    self.on_connection(conn)

  def loop(self):
    with serve(self._on_connection, self.host, self.port) as server:
      server.serve_forever()

