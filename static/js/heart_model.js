// heart_model.js

// ----------------------
// Setup Scene, Camera, Renderer
// ----------------------
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  65,
  window.innerWidth / window.innerHeight,
  0.2,
  50
);
camera.position.set(0.3, 0, 2.5);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setClearColor(0x000000, 0); // Transparent background
renderer.setSize(window.innerWidth, window.innerHeight);

// Append the renderer canvas to a container with id "webgl-container" if available,
// otherwise append it to the body.
const container = document.getElementById("webgl-container") || document.body;
container.appendChild(renderer.domElement);

// Ensure the canvas fills its container
renderer.domElement.style.width = "100%";
renderer.domElement.style.height = "100%";

// ----------------------
// Load Textures (Optional)
// ----------------------
const textureLoader = new THREE.TextureLoader();
const textures = [];
for (let i = 1; i <= 8; i++) {
  textures.push(textureLoader.load(`./models/textures/000${i}.png`));
}

// ----------------------
// Load the GLTF Heart Model
// ----------------------
const loader = new GLTFLoader();
let mixer;
let heartModel;
let lastBeatTime = 0;
const beatInterval = 1; // seconds per beat
let interpolatedBPM = 60; // For animation speed simulation
const clock = new THREE.Clock();

loader.load("./models/anim/beating-heart-v006a-animated.gltf", (gltf) => {
  heartModel = gltf.scene;
  heartModel.scale.set(0.2, 0.3, 0.3); // Adjust scale as needed
  scene.add(heartModel);

  // Apply settings to each mesh in the model
  heartModel.traverse((child) => {
    if (child.isMesh) {
      if (textures.length > 0) {
        child.material.map = textures[0];
      }
      child.castShadow = true;
      child.receiveShadow = true;
      child.material.needsUpdate = true;
      child.material.side = THREE.DoubleSide;
      child.material.flatShading = false;
      child.geometry.computeVertexNormals();
    }
  });

  // Setup animation mixer and play all available animations
  mixer = new THREE.AnimationMixer(heartModel);
  gltf.animations.forEach((clip) => {
    mixer.clipAction(clip).play();
  });

  // Add Lights to the Scene
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight1.position.set(5, 5, 5).normalize();
  directionalLight1.castShadow = true;
  scene.add(directionalLight1);

  const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight2.position.set(-15, 15, -15).normalize();
  directionalLight2.castShadow = false;
  scene.add(directionalLight2);

  const pointLight = new THREE.PointLight(0xffffff, 1, 100);
  pointLight.position.set(0, 2, 2);
  scene.add(pointLight);

  const extraDirectionalLight = new THREE.DirectionalLight(0xffffff, 8);
  extraDirectionalLight.position.set(1, 1, 1).normalize();
  scene.add(extraDirectionalLight);

  // Start the animation loop once the model is loaded
  animate();
});

// ----------------------
// Animation Loop
// ----------------------
function smoothHeartRateTransition(delta) {
  const transitionSpeed = 0.05;
  // For simulation purposes, we maintain interpolatedBPM at 60.
  interpolatedBPM += (60 - interpolatedBPM) * transitionSpeed * delta;
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  const elapsedTime = clock.getElapsedTime();

  smoothHeartRateTransition(delta);
  const speedFactor = 60 / interpolatedBPM;

  if (mixer) {
    mixer.update(delta * speedFactor);
    if (elapsedTime - lastBeatTime >= beatInterval / (interpolatedBPM / 60)) {
      lastBeatTime = elapsedTime;
      // Optional: trigger beat effects here.
    }
  }

  renderer.render(scene, camera);
}

// ----------------------
// Handle Window Resize
// ----------------------
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});