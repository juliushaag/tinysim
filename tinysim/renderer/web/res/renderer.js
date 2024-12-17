import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.165.0/three.module.min.js'
import AssetManager  from "./asset.js";
import WebSocketConnection from "./connection.js";
import Scene from "./scene.js";

const scene = new Scene()
const assets = new AssetManager("data/")
const connection = new WebSocketConnection(window.location.hostname, 5001)


const bodies = {}
let root = null

connection.register_instruction("LOAD_MESH", data => assets.load_geometry(data))

connection.register_instruction("LOAD_TEXTURE", data => assets.load_texture(data))

connection.register_instruction("LOAD_MATERIAL", data =>  assets.load_material(data))

connection.register_instruction("RESET", data => scene.clear())

connection.register_instruction("UPDATE_TRANSFORM", data => {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();

  for (const [name, transform] of Object.entries(data)) {

    const body = bodies[name]
    // console.log(body)

    // body.position.set(...transform.position);
    // body.quaternion.set(...transform.quaternion);
    // body.scale.set(...transform.scale);

  }
});

connection.register_instruction("CREATE_OBJECT", body =>  {

  const bodyObj = new THREE.Group()
  bodyObj.name = body.name

  
  bodyObj.position.set(...body.transform.position)
  bodyObj.quaternion.set(...body.transform.quaternion)
  bodies[bodyObj.name] = bodyObj


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
    
  
    const material = new THREE.MeshPhysicalMaterial({color: new THREE.Color(1, 1, 1, 1).getHex()}) 
    const visualObj = new THREE.Mesh(geometry, material)


    if (visual.mesh) {
      visualObj.visible = false
      assets.on_geometry_load(visual.mesh, geometry => {
        visualObj.geometry = geometry
        visualObj.visible = true
        visualObj.needs_update = true
      })
    }
    if (visual.material) {
      assets.on_material_load(visual.material, material => {
        visualObj.material = material
        visualObj.needs_update = true
      })
    }


    visualObj.scale.set(...visual.transform.scale)
    visualObj.position.set(...visual.transform.position)
    visualObj.quaternion.set(...visual.transform.quaternion)
    visuals.add(visualObj)

  })

  console.log(body, bodies)

  if (body.parent) bodies[body.parent].add(bodyObj)
  else {
    scene.add_object(bodyObj)
    root = bodyObj
  } 
})


connection.connect()
scene.render()