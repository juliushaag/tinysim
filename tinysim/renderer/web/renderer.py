from dataclasses import dataclass, field
from hashlib import md5
from http.server import ThreadingHTTPServer
from io import BytesIO
from threading import Thread
import time
from typing import Optional
from uuid import uuid4
import numpy as np

from scipy.spatial.transform import Rotation as R

from tinysim.renderer import RenderPrimitiveType, Renderer, RenderTransform
from tinysim.renderer.web.web_http_handler import WebRequestHandler
from tinysim.renderer.web.websocket_handler import WebSocketServer, WebSocketConnection




@dataclass(frozen=True)
class WebMesh:
  name : str
  hash : str = None
  vertex_layout : tuple = None
  normal_layout : tuple = None
  index_layout : tuple = None
  uv_layout : tuple = None

@dataclass(frozen=True)
class WebMaterial:
  name : str
  color: list[float]
  emission: float = 0.5
  specular: float = 0.5
  shininess: float = 0.5
  reflectance: float = 0.0
  texture: Optional[str] = None
  texrepeat: tuple = field(default_factory=lambda: (1, 1))

@dataclass(frozen=True)
class WebTexture:
  name : str
  hash : str
  width : int = 0
  height : int = 0
  textureType : str = "2D"

@dataclass(frozen=True)
class WebVisual:
  type : str
  mesh : str
  material : str
  transform : RenderTransform

@dataclass()
class WebObject:
  name : str
  parent : str
  transform : RenderTransform
  visuals : list[WebVisual] = field(default_factory=list)

  _dirty = False

def convert_transform(transform : RenderTransform, type = None) -> RenderTransform: 
  transform.position = transform.position[[0, 2, 1]]
  transform.position[2] *= -1
  transform.scale = transform.scale[[0, 2, 1]]


  transform.quaternion = R.from_quat(transform.quaternion, scalar_first=True).as_euler("xzy")
  transform.quaternion[2] *= -1
  transform.quaternion = R.from_euler("xyz", transform.quaternion).as_quat()
  
  if type == RenderPrimitiveType.CUBE:
    transform.scale = transform.scale * 2

  return transform

class WebRenderer(Renderer):

  def __init__(self, host="127.0.0.1", port=5000, ws_port=5001):
    
    WebRequestHandler.on_data = self.on_data_request
    self.web_server = ThreadingHTTPServer((host, port), WebRequestHandler)
    self.web_server_thread = Thread(target=self.web_server.serve_forever)
    self.web_server_thread.start()
    
    self.ws_server = WebSocketServer(host, ws_port, self.on_client_connection)
    self.ws_server_thread = Thread(target=self.ws_server.loop)
    self.ws_server_thread.start()


    self.assets = dict()

    self.meshes = dict()
    self.textures = dict()
    self.materials = dict()

    self.object_list : list[WebObject]= list()
    self.objects : dict[str, WebObject] = dict()

    self.clients : list[WebSocketConnection] = list()

    self.loaded = False

  def update(self):
    updates = dict()
    self.loaded = True
    for obj in self.object_list:
      if obj._dirty:
        updates[obj.name] = obj.transform
        obj._dirty = False

    if len(updates) == 0: return

    for client in list(self.clients):
      if not client.connected:
        self.clients.remove(client)
        del client
      else:
        client.send("UPDATE_TRANSFORM", updates)

  def on_client_connection(self, client : WebSocketConnection):

    while not self.loaded: time.sleep(0.1)

    client.send("RESET")

    for mesh in self.meshes.values():
      client.send("LOAD_MESH", mesh)

    for tex in self.textures.values():
      client.send("LOAD_TEXTURE", tex)

    for mat in self.materials.values():
      client.send("LOAD_MATERIAL", mat)

    print([obj.name for obj in self.object_list])
    for obj in self.object_list:
      client.send("CREATE_OBJECT", obj)
    
    self.clients.append(client)

    while True:
      ...

  def on_data_request(self, data_id : str):
    if data_id in self.assets:
      return self.assets[data_id]
    return None


  def create_object(self, name : str, transform : RenderTransform, parent : str = None) -> str:
    name = name or str(uuid4())

    obj = self.objects[name] = WebObject(
      name=name,
      parent=parent,
      transform=convert_transform(transform)
    )

    self.object_list.append(obj)
    
    return name

  def update_transform(self, name : str, transform : RenderTransform) -> None:
    transform = convert_transform(transform)
    self.objects[name].transform.position = transform.position
    self.objects[name].transform.quaternion = transform.quaternion
    self.objects[name]._dirty = True


  def attach_mesh(self, obj_name : str, mesh : str, material : str, transform : RenderTransform) -> None:
    transform.quaternion = R.from_quat(transform.quaternion, scalar_first=True).as_quat()

    self.objects[obj_name].visuals.append(
      WebVisual(
        type="MESH",
        material=material,
        mesh=mesh,
        transform=transform
      )
    )

  def attach_primitive(self, obj_name : RenderPrimitiveType, type : str, material : str, transform : RenderTransform) -> None:
    self.objects[obj_name].visuals.append(
      WebVisual(
        type=type,
        material=material,
        mesh=None,
        transform=convert_transform(transform, type)
      )
    )

  def create_material(
    self, 
    name : str, 
    color : np.ndarray, 
    emission : float = 0.5, 
    shininess : float = 0.5,  
    reflectance : float = 0.5,
    texture : str = None,
    texrepeat = (1, 1)
    ) -> str:

    name = name or str(uuid4())
    material = WebMaterial(
      name=name,
      color=color,
      emission=emission,
      shininess=shininess,
      reflectance=reflectance,
      texture=texture,
      texrepeat=texrepeat
    )

    self.materials[name] = material

    return name

  def create_texture(self, name : str, data : np.ndarray, width : int, height : int) -> str:
      
    bytes_data = data.tobytes()
    hash = md5(bytes_data)

    texture = WebTexture(
      name=name,
      hash=hash,
      width=width,
      height=height
    )

    self.textures[name] = texture
    self.assets[hash] = bytes_data

    return name

  def create_mesh(self, name : str, vertices : np.ndarray, indices : np.ndarray, normals : np.ndarray, uvs : np.ndarray) -> str:

    
    mesh_data = BytesIO()
    vertex_layout = mesh_data.tell(), len(vertices.flatten())
    mesh_data.write(vertices)

    index_layout = mesh_data.tell(), len(indices.flatten())
    mesh_data.write(indices)

    normal_layout = mesh_data.tell(), len(normals.flatten())
    mesh_data.write(normals)
    
    uv_layout = mesh_data.tell(), len(uvs.flatten()) if uvs else 0
    if(uvs): mesh_data.write(uvs)

    bytes_data = mesh_data.getbuffer()

    mesh = WebMesh(
      name=name,
      hash=md5(bytes_data).hexdigest(),
      vertex_layout=vertex_layout,
      index_layout=index_layout,
      normal_layout=normal_layout,
      uv_layout=uv_layout
    )

    self.meshes[name] = mesh
    self.assets[mesh.hash] = bytes_data
    return name
  
Renderer.register_backend("web", WebRenderer)