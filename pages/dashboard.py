# pages/dashboard.py
from dotenv import load_dotenv
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import os
import time
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import streamlit_antd_components as sac

# # Load environment variables from the .env file
load_dotenv()
# Import heart configuration functions from the root directory.
# (Ensure your project root is in PYTHONPATH.)
from sleep.sleeppatterns import(
    SleepPattern,
    load_sleep_data_for_week,
    calculate_average_sleep_duration,
    analyze_average_stages,
    visualize_avg_sleep_data,

)
from config import (
    update_heart_rate_data_for_day,
    plot_heart_rate_for_day,
    process_heart_rate_data,
    load_heart_rate_data_for_day,
    load_weekly_heart_rate_data,
    plot_weekly_heart_rate,
    plot_rmssd_trends,
    plot_sleep_stages,
    process_sleep_data,
    fetch_sleep_data,
    load_sleep_data,
    plot_sleep_trends,
    analyze_sleep_stage_transitions,
)
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
# --- Define a LangChain-based recommendation function ---

def get_healthcare_recommendation_langchain(current_bpm, resting_bpm, avg_bpm):
    prompt_template = """A patient is being monitored and the following heart rate statistics have been recorded:
        - Current BPM (last hour average): {current_bpm}
        - Overall average BPM: {avg_bpm}
        - Resting BPM (lowest value in the 24h period): {resting_bpm}

        Provide a concise, easy-to-understand healthcare recommendation for caregivers.
        Include recommendations for cases where the BPM is elevated, too low, or normal.
        Be clear, actionable, and avoid technical jargon.
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["current_bpm", "avg_bpm", "resting_bpm"])
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)  # Your API key should be configured in st.secrets or env variables.
    chain = LLMChain(llm=llm, prompt=prompt)
    recommendation = chain.run(current_bpm=current_bpm, avg_bpm=avg_bpm, resting_bpm=resting_bpm)
    return recommendation

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
    st.markdown("<h2 style='font-weight:bold;'>📈 Dashboard</h2>", unsafe_allow_html=True)
     # Create Container for Tabs
    st.markdown("""
    <style>
    /* This selector targets Streamlit's horizontal block columns.
       It may need adjustments if Streamlit's internal structure changes. */
    [data-testid="stHorizontalBlock"] > div {
        border: 1px solid #746f6f7d;
        padding: 20px;
        box-shadow: 1px 1px 4px rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        margin: 5px;
    }
    .st-emotion-cache-ocqkz7 {
    flex-wrap: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        tab1, tab2 = st.tabs(["🛏️ Sleep Quality Analysis", "🩸 Blood Oxygen Situration Analysis"])

        # Sleep Quality Tab
        with tab1:
            # ----- FIRST ROW: Sleep Stages & Sleep Benchmark -----
            with st.container():
                row1_col1, row1_col2 = st.columns(2)
                
                with row1_col1:
                    # Process sleep data and display sleep stages chart with total duration badges
                    sleep_data = load_sleep_data()
                    if not sleep_data:
                        sleep_data = fetch_sleep_data()
                    if sleep_data:
                        df_sleep = process_sleep_data(sleep_data)
                        plot_sleep_stages(df_sleep)
                        
                        # Calculate total duration (in seconds) per sleep stage
                        total_durations = df_sleep.groupby("level")["seconds"].sum().reset_index()
                        
                        def format_duration(sec):
                            hrs = sec // 3600
                            mins = (sec % 3600) // 60
                            secs = sec % 60
                            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
                        
                        st.markdown("###### Total Duration by Sleep Stage")
                        
                        # Define colors and icons for each sleep stage
                        sleep_colors = {
                            "wake": "#c19a6b",
                            "rem": "#b19cd9",
                            "light": "#654ea3",
                            "deep": "#000033",
                        }
                        stage_icons = {
                            "wake": "☀️",
                            "rem": "💤",
                            "light": "🌙",
                            "deep": "😴"
                        }
                        
                        # Create a row of badges styled as inline HTML
                        badge_html = "<div style='display: flex; gap: 10px; flex-wrap: wrap;'>"
                        for idx, row in total_durations.iterrows():
                            stage = row["level"]
                            duration = format_duration(row["seconds"])
                            color = sleep_colors.get(stage, "#ccc")
                            icon = stage_icons.get(stage, "")
                            badge_html += (
                                f"<span style='background-color: {color}; color: white; border-radius: 10px; "
                                "padding: 10px; font-size: 14px;'>"
                                f"{icon} {stage.capitalize()}: {duration}</span>"
                            )
                        badge_html += "</div>"
                        st.markdown(badge_html, unsafe_allow_html=True)
                
                with row1_col2:
                    if "ai_recommendation" not in st.session_state:
                        st.session_state["ai_recommendation"] = ""
                    if "ai_recommendation_generated" not in st.session_state:
                        st.session_state["ai_recommendation_generated"] = False

                    def display_ai_recommendation(details: str, recommendation: str):
                        """ Function to process AI recommendation and update session state with streaming text """
                        final_message = (
                            f"**Abnormal Sleep Stage Details:**\n\n{details}\n\n"
                            f"**Your Personalized Sleep Recommendation:**\n\n"
                        )

                        st.session_state["ai_recommendation"] = final_message
                        st.session_state["ai_recommendation_generated"] = True

                        recommendation_container = st.empty()  # Create a placeholder for streaming

                        streamed_text = ""
                        words = recommendation.split()

                        for i, word in enumerate(words):
                            streamed_text += word + " "
                            recommendation_container.markdown(
                                f'<div id="ai_recommendation_div" style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">'
                                f'{st.session_state["ai_recommendation"]}{streamed_text} ✨'
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            time.sleep(0.05)  # Adjust speed for better streaming effect
                        
                        # Store final streamed text in session state to persist
                        st.session_state["ai_recommendation"] += streamed_text

                    st.markdown("##### Sleep Benchmark")

                    if not st.session_state["ai_recommendation_generated"]:
                        # Display the sleep benchmark plot only if the AI recommendation is not generated
                        st.markdown('<div id="sleep_benchmark_plot">', unsafe_allow_html=True)
                        if not df_sleep.empty:
                            SleepPattern.plot_sleep_benchmark(df_sleep, toast_callback=display_ai_recommendation)
                        else:
                            st.warning("No sleep data available for benchmark.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Show AI Recommendation section only if the session state is active
                    if st.session_state["ai_recommendation_generated"]:
                        
                        # Display final streamed recommendation (if available)
                        st.markdown(
                            f'<div id="ai_recommendation_div" style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">'
                            f'{st.session_state["ai_recommendation"]}'
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        # Button to clear AI recommendation and restore sleep benchmark plot
                        if st.button("Clear Recommendation"):
                            st.session_state["ai_recommendation"] = ""
                            st.session_state["ai_recommendation_generated"] = False
                            st.rerun()
                
            # ----- SECOND ROW: Sleep Trends & Sleep Patterns -----
            with st.container():
                row2_col1, row2_col2 = st.columns(2)
                
                with row2_col1:
                    st.markdown("##### Sleep Trends Over Time")
                    # Arrange the inputs in one row using columns.
                    input_cols = st.columns([1, 1, 1])
                    
                    with input_cols[0]:
                        start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2025-02-15"))
                    with input_cols[1]:
                        end_date = st.date_input("📅 End Date", value=pd.to_datetime("2025-02-27"))
                    with input_cols[2]:
                        period = st.selectbox("Analysis Period", options=["daily", "weekly"], index=0)
                    
                    # Convert the dates to string format for the chart function.
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")
                    
                    # Generate and render the chart.
                    chart = plot_sleep_trends(start_date=start_date_str, end_date=end_date_str, period=period)
                    st.altair_chart(chart, use_container_width=True)
                                
                with row2_col2:
                    st.markdown("##### Sleep Patterns")
                    #analyze_sleep_stage_transitions(df_sleep)
                    week_data = load_sleep_data_for_week()
                    total_avg_duration = calculate_average_sleep_duration(week_data)
                    total_avg_each_stage_duration = analyze_average_stages(week_data)
                    visualize_avg_sleep_data(total_avg_duration, total_avg_each_stage_duration)
                   
              
        # SPO2 Analysis Tab (Placeholder)
        with tab2:
            st.subheader("SPO2 Levels Analysis (Coming Soon)")
            st.info("This feature will analyze blood oxygen saturation (SpO2) during sleep.")
            # Sleep benchmarks


elif st.session_state["menu_option"] == "heart_model":
    # Divide the page into two columns.
    col1, col2 = st.columns([3, 3])
    
    # --- Column 1: Heart Rate Data and Trends ---
    with col1:
        st.title("Heart Rate Monitoring")
        # Create three tabs for different trends.
        tabs = st.tabs(["24 Hours", "Yesterday", "One Week Trends", "Heart Rate Variability"])
        
        # --- Tab 1: 24 Hours Trends ---
        with tabs[0]:
            st.header("24 Hours Trends")
            # For 24-hour trends, we use today's date.
            date_today = datetime.now().strftime("%Y-%m-%d")
            # Download and save today's intraday data (using label "24h").
            _ = update_heart_rate_data_for_day(date_today, "24h")
            # Plot the 24-hour heart rate data.
            plot_heart_rate_for_day(date_today, "24h")
            #--------- 24 hours Heart Rate Summary --------------------------------
            # Load today's data and process it into a DataFrame.
            dataset = load_heart_rate_data_for_day(date_today, "24h")
            df_heart_rate = process_heart_rate_data(dataset)
            
            st.markdown("<h3 style=color: #FFD700;'>Heart Rate Summary</h1>", unsafe_allow_html=True)
            if df_heart_rate is not None and not df_heart_rate.empty:
                st.markdown("<div class='metric-container'>", unsafe_allow_html= True)
                avg_bpm = round(df_heart_rate["value"].mean(), 2)
                max_bpm = df_heart_rate["value"].max()
                min_bpm = df_heart_rate["value"].min()
                st.markdown(f"""
                    <div class='metric-container'>
                        <div class='metric'>📈 <br> <b>Average BPM</b> <br> <h3>{avg_bpm}</h3></div>
                        <div class='metric'>🔥 <br> <b>Max BPM</b> <br> <h3>{max_bpm}</h3></div>
                        <div class='metric'>💙 <br> <b>Min BPM</b> <br> <h3>{min_bpm}</h3></div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("No heart rate data available.")
            st.markdown("<h3 style=color: #FF4500;'>⚠️ Alerts</h3>", unsafe_allow_html=True)
            if df_heart_rate is not None and not df_heart_rate.empty:
                high_alert = df_heart_rate["value"].max() > 120
                low_alert = df_heart_rate["value"].min() < 50

                if high_alert:
                    st.warning("⚠ High Heart Rate Alert! BPM exceeds 120.")
                if low_alert:
                    st.warning("⚠ Low Heart Rate Alert! BPM below 50.")
                if not high_alert and not low_alert:
                    st.success("✅ No abnormal heart rate detected.")
            else:
                st.write("No alerts at this moment.")
        
        # --- Tab 2: Yesterday Trends ---
        with tabs[1]:
            st.header("Yesterday Trends")
            date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            _ = update_heart_rate_data_for_day(date_yesterday, "yesterday")
            plot_heart_rate_for_day(date_yesterday, "yesterday")
            yes_dataset= load_heart_rate_data_for_day(date_yesterday, "yesterday")
            df_yesterday_heart_rate = process_heart_rate_data(yes_dataset)
            #----------- Yesterday heart rate summary --------------------------------
            st.markdown("<h3 style=color: #FFD700;'>Heart Rate Summary</h1>", unsafe_allow_html=True)
            if df_yesterday_heart_rate is not None and not df_yesterday_heart_rate.empty:
                st.markdown("<div class='metric-container'>", unsafe_allow_html= True)
                avg_bpm = round(df_yesterday_heart_rate["value"].mean(), 2)
                max_bpm = df_yesterday_heart_rate["value"].max()
                min_bpm = df_yesterday_heart_rate["value"].min()
                st.markdown(f"""
                    <div class='metric-container'>
                        <div class='metric'>📈 <br> <b>Average BPM</b> <br> <h3>{avg_bpm}</h3></div>
                        <div class='metric'>🔥 <br> <b>Max BPM</b> <br> <h3>{max_bpm}</h3></div>
                        <div class='metric'>💙 <br> <b>Min BPM</b> <br> <h3>{min_bpm}</h3></div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("No heart rate data available.")
        # --- Tab 3: One Week Trends ---
        with tabs[2]:
            st.header("One Week Trends")
            weekly_data = load_weekly_heart_rate_data()
            plot_weekly_heart_rate()

            # -------------- Summary of Weekly Heart Rate Trends --------------------------------
            st.markdown("<h3 style='color: #FFD700;'>Weekly Heart Rate Summary</h3>", unsafe_allow_html=True)

            # Convert weekly_data dictionary to a DataFrame
            if weekly_data:
                data_list = []
                for date, records in weekly_data.items():
                    for record in records:
                        data_list.append({"Date": date, "Time": record["time"], "Heart Rate": record["value"]})

                df_weekly = pd.DataFrame(data_list)

                if not df_weekly.empty:
                    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                    avg_bpm = round(df_weekly["Heart Rate"].mean(), 2)
                    max_bpm = df_weekly["Heart Rate"].max()
                    min_bpm = df_weekly["Heart Rate"].min()

                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric'>📈 <br> <b>Average BPM</b> <br> <h3>{avg_bpm}</h3></div>
                            <div class='metric'>🔥 <br> <b>Max BPM</b> <br> <h3>{max_bpm}</h3></div>
                            <div class='metric'>💙 <br> <b>Min BPM</b> <br> <h3>{min_bpm}</h3></div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.write("No heart rate data available for the past week.")
            else:
                st.write("No heart rate data available.")
                    
        #----------- markdows for summary report --------------------
        st.markdown("""
            <style>
                .metric-container {
                    display: flex;
                    justify-content: space-around;
                    align-items: center;
                    margin-top: 20px;
                }
                .metric {
                    background-color: #24273A;
                    padding: 15px;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    width: 30%;
                    font-size: 18px;
                    box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.2);
                }
             </style>
        """, unsafe_allow_html=True)
        
        # Compute the current BPM from the last one hour.
        if df_heart_rate is not None and not df_heart_rate.empty:
            one_hour_ago = datetime.now() - timedelta(hours=1)
            df_last_hour = df_heart_rate[df_heart_rate.index >= one_hour_ago]
            if df_last_hour is not None and not df_last_hour.empty:
                current_bpm = round(df_last_hour["value"].mean(), 2)
            else:
                current_bpm = 60  # default value if no data in the last hour
        else:
            current_bpm = 60
        st.session_state["current_bpm"] = current_bpm
        
        # For the purpose of recommendation, assume resting BPM is the minimum value.
        resting_bpm = min_bpm if df_heart_rate is not None and not df_heart_rate.empty else 60
        
        # --- Generate personalized recommendation using LangChain ---
        recommendation = get_healthcare_recommendation_langchain(current_bpm, resting_bpm, avg_bpm)
        #--------  Heart Rate variability --------------------------------
        with tabs[3]:
            st.header("Heart Rate Varibility")
            date_today = datetime.now().strftime("%Y-%m-%d")
            plot_rmssd_trends(date_today, "24h")
    # --- Column 2: 3D Heart Model Visualization ---
    with col2:
        with st.container():
            st.subheader("❤️ Heart Model")
            current_bpm = st.session_state.get("current_bpm", 60)
            components.html(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <title>3D Heart Model</title>
                    <style>
                        body {{ margin: 0; overflow: hidden; }}
                    </style>
                    <!-- Define an import map so that bare module specifiers resolve correctly -->
                    <script type="importmap">
                    {{
                        "imports": {{
                            "three": "https://unpkg.com/three@0.150.0/build/three.module.js",
                            "three/examples/jsm/loaders/GLTFLoader.js": "https://unpkg.com/three@0.150.0/examples/jsm/loaders/GLTFLoader.js"
                        }}
                    }}
                    </script>
                </head>
                <body>
                <!-- Tooltip to display the current BPM -->
                <div id="bpm-tooltip" style="position: absolute; top: 10px; left: 10px; z-index: 9999; background-color: rgba(255,255,255,0.7); padding: 5px; border-radius: 5px; font-family: sans-serif;">
                    Current BPM: {current_bpm}
                </div>
                <div id="webgl-container"></div>
                <script type="module">
                    import * as THREE from "three";
                    import {{ GLTFLoader }} from "three/examples/jsm/loaders/GLTFLoader.js";

                    const scene = new THREE.Scene();
                    const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.2, 50);
                    camera.position.set(0.5, 0, 2.8);

                    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                    renderer.setClearColor(0x000000, 0);
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    const container = document.getElementById("webgl-container") || document.body;
                    container.appendChild(renderer.domElement);
                    renderer.domElement.style.width = "100%";
                    renderer.domElement.style.height = "100%";

                    const textureLoader = new THREE.TextureLoader();
                    const textures = [];
                    for (let i = 1; i <= 8; i++) {{
                        textures.push(textureLoader.load(`http://localhost:8000/models/textures/000${{i}}.png`));
                    }}

                    const currentBPM = {current_bpm};
                    let targetBPM = currentBPM;
                    let interpolatedBPM = currentBPM;

                    const loader = new GLTFLoader();
                    let mixer;
                    let heartModel;
                    let lastBeatTime = 0;
                    const clock = new THREE.Clock();

                    loader.load(
                        "http://localhost:8000/models/anim/beating-heart-v007a-animated.gltf",
                        (gltf) => {{
                            heartModel = gltf.scene;
                            heartModel.scale.set(0.3, 0.4, 0.4);
                            scene.add(heartModel);

                            heartModel.traverse((child) => {{
                                if (child.isMesh) {{
                                    if (textures.length > 0) {{
                                        child.material.map = textures[0];
                                    }}
                                    child.castShadow = true;
                                    child.receiveShadow = true;
                                    child.material.needsUpdate = true;
                                    child.material.side = THREE.DoubleSide;
                                    child.material.flatShading = false;
                                    child.geometry.computeVertexNormals();
                                }}
                            }});

                            mixer = new THREE.AnimationMixer(heartModel);
                            gltf.animations.forEach((clip) => {{
                                mixer.clipAction(clip).play();
                            }});

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
                        }}
                    );

                    function smoothHeartRateTransition(delta) {{
                        const transitionSpeed = 0.5;
                        interpolatedBPM += (targetBPM - interpolatedBPM) * transitionSpeed * delta;
                    }}

                    function animate() {{
                        requestAnimationFrame(animate);
                        const delta = clock.getDelta();
                        const elapsedTime = clock.getElapsedTime();

                        smoothHeartRateTransition(delta);
                        const speedFactor = interpolatedBPM / 60;
                        if (mixer) {{
                            mixer.update(delta * speedFactor);
                            if (elapsedTime - lastBeatTime >= 60 / interpolatedBPM) {{
                                lastBeatTime = elapsedTime;
                                if (heartModel) {{
                                    heartModel.scale.set(0.22, 0.33, 0.33);
                                }}
                            }}
                            if (heartModel) {{
                                heartModel.scale.lerp(new THREE.Vector3(0.2, 0.3, 0.3), 0.1);
                            }}
                        }}
                        const tooltip = document.getElementById("bpm-tooltip");
                        if (tooltip) {{
                            tooltip.innerHTML = "Current BPM: " + Math.round(interpolatedBPM);
                        }}
                        renderer.render(scene, camera);
                    }}

                    window.addEventListener("resize", () => {{
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    }});
                </script>
                </body>
                </html>
                """,
                height=300,
                scrolling=True
            )
        with st.container(border = True):
            st.markdown("### Personalized Healthcare Recommendation")
            def stream_data():
                for word in recommendation.split(" "):
                 yield word + " "
                 time.sleep(0.05)
            st.write_stream(stream_data)

elif st.session_state["menu_option"] == "respiratory_model":
    st.subheader("🫁 Respiratory Model")
    st.write("Respiratory Model visualization goes here...")
