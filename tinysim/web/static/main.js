import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.165.0/three.module.min.js'
import Scene from './scene.js'

const RenderScene = new Scene()

let scene_objects = {}

function convertPosition(x, y, z) {
  return new THREE.Vector3(x, y, z)
}

function convertQuaternion(w, x, y, z) {
  return new THREE.Vector4(x, y, z, w)
}

function convertScale(type, x, y, z) {
  let vec = convertPosition(x, y, z)
  x = vec.x
  y = vec.y
  z = vec.z
  if (type == "CYLINDER") return new THREE.Vector3(0.5 * x, 2 * y, 0.5 * z)
  else if (type == "CUBE") return new THREE.Vector3(Math.abs(x) * 2, Math.abs(y) * 2, Math.abs(z) * 2)
  else return new THREE.Vector3(x, y, z) 
}

function create_body(assets, body) {
  const bodyObj = new THREE.Group()
  bodyObj.name = body.name

  bodyObj.position.copy(convertPosition(...body.trans.pos))
  bodyObj.quaternion.copy(convertQuaternion(...body.trans.rot))
  bodyObj.scale.copy(convertPosition(...body.trans.scale))

  const visuals = new THREE.Group()
  visuals.name = "Visuals";
  bodyObj.add(visuals)

  body.visuals.forEach(visual => {

    const geometry = {
      'MESH' : () => new THREE.BoxGeometry(),
      "PLANE": () => new THREE.PlaneGeometry(),
      "SPHERE": () => new THREE.SphereGeometry(),
      "CUBE" : () => new THREE.BoxGeometry(),
      "CYLINDER" : () => new THREE.CylinderGeometry(),
      "CAPSULE" : () => new THREE.CapsuleGeometry(),
    }[visual.type]()
    
  
    const material = new THREE.MeshBasicMaterial({color: new THREE.Color(...visual.color).getHex()}) 
    const visualObj = new THREE.Mesh(geometry, material)


    if (visual.type == "MESH") {
      visualObj.visible = false
      if (visual.mesh in assets.meshes) 
        assets.meshes[visual.mesh].push(visualObj)
      else 
        assets.meshes[visual.mesh] = [visualObj]
    }

    if (visual.material) {
      if (visual.material in assets.material)
        assets.material[visual.material].push(visualObj)
      else
        assets.material[visual.material] = [visualObj]
    }


    visualObj.position.copy(convertPosition(...visual.trans.pos))
    visualObj.quaternion.copy(convertQuaternion(...visual.trans.rot))
    visualObj.scale.copy(convertScale(visual.type, ...visual.trans.scale))
    visuals.add(visualObj)
  })

  
  body.children.forEach(child => {
    const childObj = create_body(assets, child)
    bodyObj.add(childObj)
    scene_objects[child.name] = childObj
  })
  
  return bodyObj
}


let update_fn = undefined

let scene_root = undefined

async function construct_scene(data) {
  
  if (update_fn) clearInterval(update_fn);

  RenderScene.clear()
  scene_objects = {}

  let asset_request = {
    meshes : {},
    material : {}
  }

  // Build scene
  scene_root = create_body(asset_request, data.root)
  RenderScene.add_object(scene_root)

  scene_root.rotation.x = -Math.PI * 0.5
  scene_root.rotation.z = Math.PI * 2


  // // Center scene
  const center = new THREE.Vector3();
  const boundingBox = new THREE.Box3();
  
  boundingBox.setFromObject(scene_root);
  boundingBox.getCenter(center);
  scene_root.position.set(-center.x, 0.1 , -center.z)

  console.log("Loaded scene", scene_root, data)
  
  // update_fn = setInterval(update_scene, 200); 

  // Post load required assets

  // Contruct meshes
  var meshes = Object.entries(asset_request.meshes).map(entry => { 
    const [mesh_name, scene_objs] = entry
    const mesh_info = data.meshes.find(mesh_entry => mesh_entry.name == mesh_name)

    return fetch("/data/" + mesh_info.hash)
      .then(req => req.blob())
      .then(req => req.arrayBuffer())
      .then(mesh_data => {
        let mesh = construct_mesh(mesh_info, mesh_data)
        scene_objs.forEach(obj => {
          obj.geometry.dispose()
          obj.geometry = mesh
          obj.visible = true
          obj.needsUpdate = true
          console.log(obj)
        }
      )}
    )
  })
  
  var materials = Object.entries(asset_request.material).map(entry => {

    const [mat_name, scene_objs] = entry
    const mat_info = data.materials.find(mat_entry => mat_entry.name == mat_name)

    return construct_material(mat_info, data.textures).then(
      material => scene_objs.forEach(obj => {
        obj.material = material
        obj.material.needsUpdate = true
      })
    )

  })

  Promise.all(meshes, materials)
  
}

async function update_scene() {

  clearInterval(update_fn);

  var data = await (await fetch("/scene_state")).json()
  
  if ("updateData" in data) {

    for (var [name, value] of Object.entries(data.updateData)) {
      const obj = scene_objects[name]
      if (!obj) return

      const worldposition = convertLHtoRH(value[0], value[1], value[2]).add(scene_root.position)
      const newquat = convertQuaternionLHtoRH(value[3], value[4], value[5], value[6]).multiply(scene_root.quaternion)
      
      const localposition = obj.parent.worldToLocal(worldposition)
      obj.position.copy(localposition);
      
      const worldQuaternion = new THREE.Quaternion();
      obj.parent.getWorldQuaternion(worldQuaternion).invert()
      obj.quaternion.copy(worldQuaternion).multiply(newquat)
    }
  }

  update_fn = setInterval(update_scene, 100); 
}

function construct_mesh(mesh, data) {
  
  const geometry = new THREE.BufferGeometry();

  const indices = new Uint32Array(data, mesh.indicesLayout[0], mesh.indicesLayout[1]);
  const vertices = new Float32Array(data, mesh.verticesLayout[0], mesh.verticesLayout[1]);
  const normals = new Float32Array(data, mesh.normalsLayout[0], mesh.normalsLayout[1]);
  const uvs = new Float32Array(data, mesh.uvLayout[0], mesh.uvLayout[1]);

  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  if (uvs.length > 0) {
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
  }

  return geometry  
}

async function construct_material(material, textures) {

  const mat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(material.color[0], material.color[1], material.color[2]),
    emissive: new THREE.Color(material.color[0] * material.emissive, material.color[1] * material.emissive, material.color[2] * material.emissive),
    roughness: 1.0 - material.shininess, // Roughness is inverse of shininess
    metalness: material.reflectance,
    specularIntensity : material.specular,
  })

  if (material.texture) {
    const texture = textures.find(tex => tex.name == material.texture)
    
    var data = await (await (await fetch("/data/" + hash)).blob()).arrayBuffer()
    const tex = construct_texture(texture, data)
    scene_objects.forEach(obj => {
      obj.material.map = tex
      obj.material.needsUpdate = true
    })
  
  }

  return mat
}

function construct_texture(texture, data) {
  const rgbData = new Uint8Array(data); // Assuming data is your RGB pixel data
  const rgbaData = new Uint8Array(rgbData.length + (rgbData.length / 3));

  for (let i = 0; i < rgbData.length; i++) {
      rgbaData[i * 4] = rgbData[i * 3];
      rgbaData[i * 4 + 1] = rgbData[i * 3 + 1];
      rgbaData[i * 4+ 2] = rgbData[i * 3 + 2];
      rgbaData[i * 4+ 3] = 255; // Set alpha to 255 (fully opaque)
  }
  
  var tex = new THREE.DataTexture(rgbaData, texture.height, texture.width, THREE.RGBAFormat)
  tex.flipY = false
  tex.needsUpdate = true
  return tex
}

let current_id = 0

let scene_id_updater = setInterval(scene_id_update, 1000)

async function scene_id_update() {

  clearInterval(scene_id_updater)
  
  var response = await fetch("/scene_id")
  var new_id = await response.text()

  
  if (current_id != new_id) {
    current_id = new_id
    var scene = await (await fetch("/scene_data")).json()
    await construct_scene(scene)
  }
  
  scene_id_updater = setInterval(scene_id_update, 1000)
}

RenderScene.render()