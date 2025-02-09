# pages/dashboard.py
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os

# Import heart configuration functions from the root directory.
# If you encounter an import error, ensure your project root is in PYTHONPATH.
from config import (
    update_heart_rate_data_for_day,
    plot_heart_rate_for_day,
    update_heart_rate_data_for_range,
    plot_heart_rate_for_range,
    process_heart_rate_data,
    load_heart_rate_data_for_day,
)

# Redirect to login page if not authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.switch_page("main.py")  # Redirect back to login

# Sidebar Menu
with st.sidebar:
    st.markdown("### 📌 Navigation")
    st.markdown("---")

    if "menu_option" not in st.session_state:
        st.session_state["menu_option"] = "trends"  # Default page

    if st.button("📈 Trends", key="trends"):
        st.session_state["menu_option"] = "trends"

    if st.button("❤️ Heart Model", key="heart_model"):
        st.session_state["menu_option"] = "heart_model"

    if st.button("🫁 Respiratory Model", key="respiratory_model"):
        st.session_state["menu_option"] = "respiratory_model"

    st.markdown("---")

    if st.button("🚪 Logout", key="logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.switch_page("main.py")  # Redirect to login

# Render the Selected Page Content
if st.session_state["menu_option"] == "trends":
    st.subheader("📈 Trends")
    st.write("Trends and analytics visualization goes here...")

elif st.session_state["menu_option"] == "heart_model":
    # Divide the page into two columns.
    col1, col2 = st.columns([3, 3])
    
    # --- Column 1: Heart Rate Data and Trends ---
    with col1:
        st.title("Heart Rate Monitoring")
        # Create three tabs for different trends.
        tabs = st.tabs(["24 Hours", "Yesterday", "One Week Trends"])
        
        # --- Tab 1: 24 Hours Trends ---
        with tabs[0]:
            st.header("24 Hours Trends")
            # For 24-hour trends, we use today's date.
            date_today = datetime.now().strftime("%Y-%m-%d")
            # Download and save today's intraday data (using label "24h").
            _ = update_heart_rate_data_for_day(date_today, "24h")
            # Plot the 24-hour heart rate data.
            plot_heart_rate_for_day(date_today, "24h")

        # --- Tab 2: Yesterday Trends ---
        with tabs[1]:
            st.header("Yesterday Trends")
            # Use yesterday's date.
            date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            _ = update_heart_rate_data_for_day(date_yesterday, "yesterday")
            plot_heart_rate_for_day(date_yesterday, "yesterday")
        
        # --- Tab 3: One Week Trends ---
        with tabs[2]:
            st.header("One Week Trends")
            # Define the week as from 7 days ago to yesterday.
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            _ = update_heart_rate_data_for_range(start_date, end_date)
            plot_heart_rate_for_range(start_date, end_date)
        # Display summary statistics.
        # Then load the data from the saved JSON file
        dataset = load_heart_rate_data_for_day(date_today, "24h")
        df_heart_rate = process_heart_rate_data(dataset)
        st.subheader("📝 Summary")
        if df_heart_rate is not None and not df_heart_rate.empty:
            avg_bpm = round(df_heart_rate["value"].mean(), 2)
            max_bpm = df_heart_rate["value"].max()
            min_bpm = df_heart_rate["value"].min()
            st.write(f"**Average BPM:** {avg_bpm}")
            st.write(f"**Max BPM:** {max_bpm}")
            st.write(f"**Min BPM:** {min_bpm}")
        else:
            st.write("No heart rate data available.")

        # Display alerts based on heart rate thresholds.
        st.subheader("⚠ Alerts")
        if df_heart_rate is not None and not df_heart_rate.empty:
            high_alert = df_heart_rate["value"].max() > 120  # Example threshold
            low_alert = df_heart_rate["value"].min() < 50    # Example threshold

            if high_alert:
                st.warning("⚠ High Heart Rate Alert! BPM exceeds 120.")
            if low_alert:
                st.warning("⚠ Low Heart Rate Alert! BPM below 50.")
            if not high_alert and not low_alert:
                st.success("✅ No abnormal heart rate detected.")
        else:
            st.write("No alerts at this moment.")

    # --- Column 2: 3D Heart Model Visualization ---
    with col2:
        st.subheader("❤️ Heart Model")
        components.html(
            """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8" />
                <title>3D Heart Model</title>
                <style>
                    body { margin: 0; overflow: hidden; }
                </style>
                <!-- Define an import map so that bare module specifiers resolve correctly -->
                <script type="importmap">
                {
                    "imports": {
                        "three": "https://unpkg.com/three@0.150.0/build/three.module.js",
                        "three/examples/jsm/loaders/GLTFLoader.js": "https://unpkg.com/three@0.150.0/examples/jsm/loaders/GLTFLoader.js"
                    }
                }
                </script>
            </head>
            <body>
            <div id="webgl-container"></div>
            <script type="module">
                import * as THREE from "three";
                import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

                // Set up the scene, camera, and renderer.
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.2, 50);
                camera.position.set(0.5, 0, 2.8);

                const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
                renderer.setClearColor(0x000000, 0);
                renderer.setSize(window.innerWidth, window.innerHeight);
                const container = document.getElementById("webgl-container") || document.body;
                container.appendChild(renderer.domElement);
                renderer.domElement.style.width = "100%";
                renderer.domElement.style.height = "100%";

                // Load textures.
                const textureLoader = new THREE.TextureLoader();
                const textures = [];
                for (let i = 1; i <= 8; i++) {
                    textures.push(textureLoader.load(`http://localhost:8000/models/textures/000${i}.png`));
                }

                // Set up random heart rate data.
                const randomHeartRates = Array.from({ length: 100 }, () =>
                    Math.floor(Math.random() * 41) + 60
                );
                let heartRateIndex = 0;
                let targetBPM = randomHeartRates[heartRateIndex];
                let interpolatedBPM = targetBPM;

                // Load the GLTF Heart Model.
                const loader = new GLTFLoader();
                let mixer;
                let heartModel;
                let lastBeatTime = 0;
                const clock = new THREE.Clock();

                loader.load(
                    "http://localhost:8000/models/anim/beating-heart-v007a-animated.gltf",
                    (gltf) => {
                        heartModel = gltf.scene;
                        heartModel.scale.set(0.3, 0.4, 0.4);
                        scene.add(heartModel);

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

                        mixer = new THREE.AnimationMixer(heartModel);
                        gltf.animations.forEach((clip) => {
                            mixer.clipAction(clip).play();
                        });

                        // Add lighting.
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

                        animate();
                    }
                );

                function smoothHeartRateTransition(delta) {
                    const transitionSpeed = 0.5;
                    interpolatedBPM += (targetBPM - interpolatedBPM) * transitionSpeed * delta;
                }

                function animate() {
                    requestAnimationFrame(animate);
                    const delta = clock.getDelta();
                    const elapsedTime = clock.getElapsedTime();

                    smoothHeartRateTransition(delta);
                    const speedFactor = interpolatedBPM / 60;
                    if (mixer) {
                        mixer.update(delta * speedFactor);
                        if (elapsedTime - lastBeatTime >= 60 / interpolatedBPM) {
                            lastBeatTime = elapsedTime;
                            heartRateIndex = (heartRateIndex + 1) % randomHeartRates.length;
                            targetBPM = randomHeartRates[heartRateIndex];
                            if (heartModel) {
                                heartModel.scale.set(0.22, 0.33, 0.33);
                            }
                        }
                        if (heartModel) {
                            heartModel.scale.lerp(new THREE.Vector3(0.2, 0.3, 0.3), 0.1);
                        }
                    }
                    renderer.render(scene, camera);
                }

                window.addEventListener("resize", () => {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                });
            </script>
            </body>
            </html>
            """,
            height=300,
            scrolling=True
        )

elif st.session_state["menu_option"] == "respiratory_model":
    st.subheader("🫁 Respiratory Model")
    st.write("Respiratory Model visualization goes here...")
