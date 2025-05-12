import pandas as pd
import altair as alt
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import streamlit_antd_components as sac
import glob
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import time
from streamlit_echarts import st_echarts
import plotly.graph_objects as go

class SleepPattern:
    
    @staticmethod
    def plot_sleep_benchmark(df_sleep, toast_callback=None):
        """
        Plot the sleep benchmark chart. If abnormalities are detected,
        a "Generate AI Recommendation" button is displayed. When clicked,
        the provided toast_callback is invoked with abnormal details and
        the AI recommendation.
        """
        # --- 1) Aggregate total time (in seconds) by stage ---
        grouped = df_sleep.groupby("level")["seconds"].sum().reset_index()
        total_seconds = grouped["seconds"].sum()
        if total_seconds == 0:
            st.warning("No valid sleep durations found.")
            return

        # --- 2) Map levels to nicer labels, colors, typical ranges ---
        level_mapping = {
            "wake": "Awake",
            "rem": "REM",
            "light": "Light",
            "deep": "Deep"
        }
        sleep_colors = {
            "Awake": "#c19a6b",
            "REM": "#b19cd9",
            "Light": "#654ea3",
            "Deep": "#000033"
        }
        typical_ranges = {
            "Awake": (10, 20),
            "REM": (15, 25),
            "Light": (40, 60),
            "Deep": (10, 20)
        }

        # --- Build DataFrame for chart ---
        data = []
        for _, row in grouped.iterrows():
            original_level = row["level"]
            stage = level_mapping[original_level]
            secs = row["seconds"]
            user_minutes = secs // 60
            user_percentage = round((secs / total_seconds) * 100)
            tmin, tmax = typical_ranges[stage]
            data.append({
                "stage": stage,
                "user_percentage": user_percentage,
                "user_minutes": user_minutes,
                "typical_min": tmin,
                "typical_max": tmax
            })

        df_bench = pd.DataFrame(data)

        # --- 2a) Add alert information for sleep abnormalities ---
        df_bench["alert"] = df_bench.apply(
            lambda row: "" if (row["user_percentage"] >= row["typical_min"] and row["user_percentage"] <= row["typical_max"])
            else ("Low" if row["user_percentage"] < row["typical_min"] else "High"),
            axis=1
        )

        abnormal_stages = df_bench[df_bench["alert"] != ""]

        if not abnormal_stages.empty and not st.session_state.get("ai_recommendation_generated", False):
            # Build detailed abnormal values string
            details = "\n".join(
                f"Stage: {row['stage']} – Duration: {row['user_minutes']} min, Percentage: {row['user_percentage']}% (Expected: {row['typical_min']}% to {row['typical_max']}%)"
                for row in abnormal_stages.to_dict("records")
            )
            st.warning("Sleep abnormalities detected:\n" + details)

            # Button to generate AI recommendation
            if st.button("Generate Care Recommendation"):
                recommendation = SleepPattern.get_ai_recommendation(details)
                if toast_callback:
                    toast_callback(details, recommendation)
                st.session_state["ai_recommendation_generated"] = True
                st.rerun()

        # --- 3) Sort by a custom order ---
        stage_order = ["Awake", "REM", "Light", "Deep"]
        df_bench["stage"] = pd.Categorical(df_bench["stage"], categories=stage_order, ordered=True)
        df_bench = df_bench.sort_values("stage")

        # --- 4) Create the Altair chart ---
        typical_range_chart = alt.Chart(df_bench).mark_bar(
            stroke='gray', strokeDash=[4, 4], fillOpacity=0, strokeWidth=1.5
        ).encode(
            x='typical_min:Q',
            x2='typical_max:Q',
            y=alt.Y('stage:N', sort=stage_order),
        )

        user_bar_chart = alt.Chart(df_bench).mark_bar().encode(
            x=alt.X('user_percentage:Q', title='Percentage (%)', scale=alt.Scale(domain=[0, 100])),

            y=alt.Y('stage:N', sort=stage_order),
            color=alt.Color('stage:N',
                            scale=alt.Scale(domain=list(sleep_colors.keys()),
                                            range=list(sleep_colors.values())),
                            legend=None)
        )
        
        text_labels = alt.Chart(df_bench).mark_text(
            align='left', baseline='middle', dx=5, color="grey"
        ).encode(
            x='user_percentage:Q',
            y=alt.Y('stage:N', sort=stage_order),
            text=alt.Text('label:N')
        ).transform_calculate(
            label='datum.stage + " • " + datum.user_percentage + "%" + " • " + datum.user_minutes + " min" + (datum.alert ? " ⚠ " + datum.alert : "")'
        )

        chart = alt.layer(typical_range_chart, user_bar_chart, text_labels).properties(width=500, height=180)
        st.altair_chart(chart, use_container_width=True)

    @staticmethod
    def get_ai_recommendation(abnormal_details: str) -> str:
        """
        Generate a structured AI recommendation with enforced bullet-point formatting.
        """

        # Define a structured prompt template
        prompt_template = """A caregiver is monitoring a patient’s sleep, and the following abnormalities were detected:

        {abnormal_details}

        Analyze these against standard norms and provide **concise, meaningful recommendations for caregivers**.
        Ensure **a maximum of 4 bullet points**, each addressing a specific abnormality with practical advice.

        ### **Format your response EXACTLY like this:**
        
        - **[First abnormality]** → [Short recommendation]  

        - **[Second abnormality]** → [Short recommendation]  

        - **[Third abnormality]** → [Short recommendation]  

        - **[Fourth abnormality]** → [Short recommendation]  

        🚨 **DO NOT** include any introduction, summary, or extra text. **ONLY provide bullet points in the format above.**
        """

        # Create a prompt template with input variable
        prompt = PromptTemplate(template=prompt_template, input_variables=["abnormal_details"])
        
        try:
            # Call GPT-4 model with the template
            llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)
            chain = LLMChain(llm=llm, prompt=prompt)
            recommendation = chain.run(abnormal_details=abnormal_details).strip()

            # **Post-processing to enforce formatting**
            # Ensure each bullet point starts with "- **" and has correct spacing
            recommendation = re.sub(r"\n\s*-\s\*\*", "\n\n- **", recommendation)  # Ensure line breaks before bullets

            return recommendation

        except Exception as e:
            return f"Error generating recommendation: {e}"
        

# ================================ weekly sleep analysis ================================

#----------- Visualization functions --------------------------------
def load_sleep_data_for_week(directory="static/data/sleep"):
    """
    Loads sleep data for the past 7 days (today and the previous 6 days) from JSON files.
    If no recent files are found, loads data from the latest available file and the previous 6 files by date.

    Returns:
        pd.DataFrame: DataFrame containing sleep data.
    """
    def parse_date_from_filename(filename):
        try:
            return datetime.strptime(filename.split("_")[0], "%Y-%m-%d")
        except ValueError:
            return None

    # Get all available sleep data files
    all_files = [
        f for f in os.listdir(directory)
        if f.endswith("_sleep.json") and parse_date_from_filename(f)
    ]
    
    # Sort files by date (newest to oldest)
    sorted_files = sorted(
        all_files,
        key=lambda f: parse_date_from_filename(f),
        reverse=True
    )

    # Attempt to load today's data and previous 6 days
    today = datetime.today()
    target_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    target_files = [f"{date}_sleep.json" for date in target_dates]

    data_list = []
    loaded_files = set()

    for file_name in target_files:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                json_data = json.load(f)
                if "sleep" in json_data and json_data["sleep"]:
                    record = json_data["sleep"][0]
                    date_of_sleep = record.get("dateOfSleep", file_name.split("_")[0])
                    minutes_asleep = record.get("minutesAsleep", 0)
                    levels = record.get("levels", {}).get("summary", {})
                    data_list.append({
                        "date": date_of_sleep,
                        "minutesAsleep": minutes_asleep,
                        "deep": levels.get("deep", {}).get("minutes", 0),
                        "light": levels.get("light", {}).get("minutes", 0),
                        "rem": levels.get("rem", {}).get("minutes", 0),
                        "wake": levels.get("wake", {}).get("minutes", 0)
                    })
                    loaded_files.add(file_name)

    # If not enough data loaded, use fallback: last 7 available files
    if len(data_list) < 7:
        print("Not enough recent data found. Using last 7 available files instead.")
        data_list = []
        for file_name in sorted_files[:7]:
            if file_name not in loaded_files:
                file_path = os.path.join(directory, file_name)
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                    if "sleep" in json_data and json_data["sleep"]:
                        record = json_data["sleep"][0]
                        date_of_sleep = record.get("dateOfSleep", file_name.split("_")[0])
                        minutes_asleep = record.get("minutesAsleep", 0)
                        levels = record.get("levels", {}).get("summary", {})
                        data_list.append({
                            "date": date_of_sleep,
                            "minutesAsleep": minutes_asleep,
                            "deep": levels.get("deep", {}).get("minutes", 0),
                            "light": levels.get("light", {}).get("minutes", 0),
                            "rem": levels.get("rem", {}).get("minutes", 0),
                            "wake": levels.get("wake", {}).get("minutes", 0)
                        })

    # Sort by date before returning
    data_list = sorted(data_list, key=lambda x: x["date"])
    return pd.DataFrame(data_list)
def calculate_average_sleep_duration(df):
    if df.empty:
        return 0
    return df["minutesAsleep"].mean()

def analyze_average_stages(df):
    if df.empty:
        return {"deep": 0, "light": 0, "rem": 0, "wake": 0}
    
    avg_deep  = df["deep"].mean()
    avg_light = df["light"].mean()
    avg_rem   = df["rem"].mean()
    avg_wake  = df["wake"].mean()
    return {"deep": avg_deep, "light": avg_light, "rem": avg_rem, "wake": avg_wake}

def visualize_avg_sleep_data(avg_total_sleep, avg_stages):
    # Round all float values to the nearest integer
    total_val = int(round(avg_total_sleep))
    deep_val  = int(round(avg_stages.get("deep", 0)))
    light_val = int(round(avg_stages.get("light", 0)))
    rem_val   = int(round(avg_stages.get("rem", 0)))
    wake_val  = int(round(avg_stages.get("wake", 0)))

    # Convert total_val from minutes to H:mm format
    total_hours = total_val // 60
    total_minutes_remaining = total_val % 60
    total_str = f"{total_hours}:{total_minutes_remaining:02d} hr"
    # Determine the maximum gauge range (add buffer so largest arc doesn't fill entire ring)
    max_val = max(total_val, deep_val, light_val, rem_val, wake_val)
    gauge_max = int(round(max_val * 1.2)) if max_val > 0 else 100

    # Build data array with five arcs
    data_items = [
        {
            "value": total_val,
            "name": "Sleep per Day (avg)",
            "title": {"offsetCenter": ["0%", "-60%"]},
            "detail": {"offsetCenter": ["0%", "-45%"], "formatter": total_str},
            "itemStyle": {"color": "#4c8aff"}  # Arc color
        },
        {
            "value": deep_val,
            "name": "Deep",
            "title": {"offsetCenter": ["0%", "-30%"]},
            "detail": {"offsetCenter": ["0%", "-17%"]},
            "itemStyle": {"color": "#673ab7"}
        },
        {
            "value": light_val,
            "name": "Light",
            "title": {"offsetCenter": ["0%", "0%"]},
            "detail": {"offsetCenter": ["0%", "11%"]},
            "itemStyle": {"color": "#00c853"}
        },
        {
            "value": rem_val,
            "name": "REM",
            "title": {"offsetCenter": ["0%", "25%"]},
            "detail": {"offsetCenter": ["0%", "35%"]},
            "itemStyle": {"color": "#ffa726"}
        },
        {
            "value": wake_val,
            "name": "Wake",
            "title": {"offsetCenter": ["0%", "50%"]},
            "detail": {"offsetCenter": ["0%", "60%"]},
            "itemStyle": {"color": "#ff5252"}
        }
    ]

    # Configure the ring gauge
    option = {
        "tooltip": {
            # {a} => series name, {b} => data name, {c} => data value
            "formatter": "{b} : {c} min"
        },
        "series": [
            {
                "name": "Sleep Data",
                "type": "gauge",
                "min": 0,
                "max": gauge_max,
                "startAngle": 90,
                "endAngle": -270,
                "pointer": {"show": False},
                "progress": {
                    "show": True,
                    "overlap": False,  # arcs placed side-by-side around ring
                    "roundCap": True,
                    "clip": False
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 50,
                        "color": [[1, "#E9E9E9"]]  # background ring color
                    }
                },
                "splitLine": {"show": False},
                "axisTick": {"show": False},
                "axisLabel": {"show": False},
                "data": data_items,
                "title": {
                    "fontSize": 12
                },
                "detail": {
                    "width": 40,
                    "height": 10,
                    "fontSize": 12,
                    "color": "auto",
                    "borderColor": "auto",
                    "borderRadius": 40,
                    "borderWidth": 1,
                    "formatter": "{value} min"  # display numeric value in ring
                }
            }
        ]
    }

    st_echarts(options=option, height="500px")

def calculate_sleep_efficiency_current_day(sleep_data) -> float:
    """
    Loads the current day's sleep JSON file (named as YYYY-MM-DD_sleep.json) and calculates sleep efficiency.
    
    Sleep Efficiency (%) = (totalMinutesAsleep / totalTimeInBed) * 100

    Returns the efficiency percentage, or None if data is missing or an error occurs.
    """   

    try:
    
        # Navigate to the summary object
        summary = sleep_data.get("summary", {})
        total_minutes_asleep = summary.get("totalMinutesAsleep")
        total_time_in_bed = summary.get("totalTimeInBed")
        
        if total_minutes_asleep is None or total_time_in_bed is None or total_time_in_bed == 0:
            st.error("Insufficient sleep summary data.")
            return None
        
        efficiency = (total_minutes_asleep / total_time_in_bed) * 100
        return efficiency
    except Exception as e:
        st.error(f"Error processing sleep data: {e}")
        return None
    
def analyze_sleep_quality(efficiency: float):
    
    # Determine quality based on efficiency
    if efficiency >= 85:
        quality = "Normal (Good sleep efficiency)"
        message = "Your sleep efficiency is normal."
        st.success(f"{quality}: {message}")
    elif efficiency >= 75:
        quality = "Mildly reduced sleep efficiency"
        message = "Your sleep efficiency is mildly reduced. Consider reviewing your sleep habits."
        st.warning(f"{quality}: {message}")
    else:
        quality = "Poor sleep efficiency (Indicative of sleep disturbances or disorders)"
        message = "Your sleep efficiency is poor. You may want to consult a professional or review your sleep environment."
        st.error(f"{quality}: {message}")

    # Create a gauge chart using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=efficiency,
        number={'suffix': "%"},
        title={'text': "Sleep Efficiency"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 75], 'color': "red"},
                {'range': [75, 85], 'color': "yellow"},
                {'range': [85, 100], 'color': "green"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def calculate_sleep_latency_current_day(sleep_data) -> float:
   
    try:
        # Assume sleep data is under the "sleep" key (take the first sleep record)
        sleep_records = sleep_data.get("sleep", [])
        if not sleep_records:
            st.error("No sleep records found in today's data.")
            return None
        
        sleep_record = sleep_records[0]
        start_time_str = sleep_record.get("startTime")
        if not start_time_str:
            st.error("No startTime found in sleep record.")
            return None
        
        # Convert startTime to a datetime object
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        
        # Retrieve levels data from the sleep record
        levels_data = sleep_record.get("levels", {}).get("data", [])
        if not levels_data:
            st.error("No levels data found in sleep record.")
            return None
        
        # Find the first event in levels that is not "wake"
        sleep_onset_time = None
        for event in levels_data:
            level = event.get("level", "").lower()
            if level != "wake":
                dt_str = event.get("dateTime")
                if dt_str:
                    sleep_onset_time = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    break
        
        if sleep_onset_time is None:
            st.error("Could not determine sleep onset time from levels data.")
            return None
        
        latency_seconds = (sleep_onset_time - start_time).total_seconds()
        return latency_seconds
    
    except Exception as e:
        st.error(f"Error processing sleep data: {e}")
        return None
    
def count_awakenings_current_day(sleep_data) -> int:
   
    try:
        
        # Assume the sleep records are under the "sleep" key; take the first record.
        sleep_records = sleep_data.get("sleep", [])
        if not sleep_records:
            st.error("No sleep records found for today.")
            return None
        
        sleep_record = sleep_records[0]
        levels_data = sleep_record.get("levels", {}).get("data", [])
        if not levels_data:
            st.error("No levels data found in today's sleep record.")
            return None
        
        # Identify sleep onset: find the first event where level is not "wake".
        first_sleep_index = None
        for i, event in enumerate(levels_data):
            if event.get("level", "").lower() != "wake":
                first_sleep_index = i
                break
        
        if first_sleep_index is None:
            # No sleep detected.
            return 0
        
        # Count transitions to "wake" after sleep onset.
        count = 0
        prev_level = levels_data[first_sleep_index].get("level", "").lower()
        for event in levels_data[first_sleep_index+1:]:
            curr_level = event.get("level", "").lower()
            if curr_level == "wake" and prev_level != "wake":
                count += 1
            prev_level = curr_level
        return count
        
    except Exception as e:
        st.error(f"Error processing sleep data: {e}")
        return None
    
#============  Sleep transion and fregramentation  ===============
def analyze_sleep_stage_transitions(df):
    """
    Analyzes sleep stage transitions and fragmentation from processed sleep data,
    and displays both visualizations and plain language interpretations in an info alert.
    
    **Algorithm:**
    
    1. Transition Probability Matrix:
       - **Sort:** Order the sleep data by start time.
       - **Shift:** Create a new column "next_stage" by shifting the 'level' column.
       - **Count:** Use pd.crosstab() to count transitions from one stage to the next.
       - **Normalize:** Divide each row by its sum to obtain the probability of transitioning 
         from a given stage to another.
       
    2. Fragmentation Index:
       - **Total Transitions:** Compute the number of transitions (number of segments minus one).
       - **Duration:** Calculate the total sleep duration (in seconds) and convert it to hours.
       - **Transitions per Hour:** Divide the total transitions by the total sleep duration (in hours).
       - **Pattern Count:** Loop through the segments to count instances of A → B → A (where a stage is interrupted 
         briefly by another stage and then returns), which indicates sleep fragmentation.
    
    **Visualizations:**
      - A heatmap displays the transition probability matrix.
      - A bar chart shows the fragmentation metrics.
      
    **Plain Language Interpretation:**
      An info alert explains the results in simple terms so non-technical users can understand what the metrics imply about their sleep quality.
    
    Parameters:
      df: DataFrame from process_sleep_data() with at least:
          - 'level': Sleep stage (e.g., "wake", "rem", "light", "deep")
          - 'dateTime': Start time of each segment (datetime)
          - 'seconds': Duration (in seconds) of each segment
    """
    # --- Compute Transition Probability Matrix ---
    df_sorted = df.sort_values("dateTime").reset_index(drop=True)
    df_sorted["next_stage"] = df_sorted["level"].shift(-1)
    
    # Count transitions from each stage to the next
    transition_counts = pd.crosstab(df_sorted["level"], df_sorted["next_stage"])
    # Normalize counts to obtain probabilities (row-wise)
    transition_prob = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    
    # Prepare data for heatmap visualization
    df_heat = transition_prob.reset_index().melt(
        id_vars="level", var_name="next_stage", value_name="probability"
    )
    
    heatmap = alt.Chart(df_heat).mark_rect().encode(
        x=alt.X("next_stage:N", title="To Stage"),
        y=alt.Y("level:N", title="From Stage"),
        color=alt.Color("probability:Q", scale=alt.Scale(scheme="blues"), title="Probability"),
        tooltip=[
            alt.Tooltip("level:N", title="From Stage"),
            alt.Tooltip("next_stage:N", title="To Stage"),
            alt.Tooltip("probability:Q", title="Probability", format=".2f")
        ]
    ).properties(
        title="Transition Probability Matrix",
        width=300,
        height=300
    )
    
    # --- Compute Fragmentation Metrics ---
    total_transitions = len(df_sorted) - 1
    total_duration_seconds = df_sorted["seconds"].sum()
    total_duration_hours = total_duration_seconds / 3600.0 if total_duration_seconds > 0 else 0
    transitions_per_hour = total_transitions / total_duration_hours if total_duration_hours > 0 else 0
    
    # Count A → B → A patterns (indicator of fragmentation)
    pattern_count = 0
    for i in range(len(df_sorted) - 2):
        stage_a = df_sorted.iloc[i]["level"]
        stage_b = df_sorted.iloc[i+1]["level"]
        stage_a_next = df_sorted.iloc[i+2]["level"]
        if stage_a == stage_a_next and stage_a != stage_b:
            pattern_count += 1
    
    # Prepare fragmentation metrics data for a bar chart
    frag_data = pd.DataFrame({
        "Metric": ["Transitions per Hour", "A→B→A Patterns"],
        "Value": [transitions_per_hour, pattern_count]
    })
    
    frag_chart = alt.Chart(frag_data).mark_bar().encode(
        x=alt.X("Metric:N", title="Fragmentation Metric"),
        y=alt.Y("Value:Q", title="Value"),
        color=alt.Color("Metric:N", scale=alt.Scale(range=["#654ea3", "#c19a6b"]))
    ).properties(
        title="Sleep Fragmentation Metrics",
        width=300,
        height=300
    )
    
    # --- Display Visualizations ---
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(heatmap, use_container_width=True)
    with col2:
        st.altair_chart(frag_chart, use_container_width=True)
    
    # --- Plain Language Interpretation in an Info Alert ---
    interpretation = (
        "**Transition Probability Matrix:**\n\n"
        "This heatmap shows how frequently the older adults sleep moves from one stage to another. "
        "High probabilities along expected paths (for example, from Light to REM or REM to Deep) indicate a stable, smooth sleep cycle. "
        "If you observe frequent transitions from Light or REM directly to Awake, it may suggest disruptions in your elderly loved one's sleep.\n\n"
        "**Fragmentation Metrics:**\n\n"
        f"- **Transitions per Hour:** {transitions_per_hour:.2f}. A lower value indicates more consolidated sleep, "
        "while a higher value suggests frequent interruptions.\n"
        f"- **A → B → A Patterns:** {pattern_count}. This metric counts how often the an older adult briefly switch away from a sleep stage and then return to it, indicating possible sleep fragmentation.\n\n"
        "Overall, fewer transitions and less fragmentation are signs of stable, restorative sleep. "
        "If the elderly metrics indicate high fragmentation or many unexpected transitions, consider reviewing the sleep environment or habits, and consult a healthcare provider if needed."
    )
    with st.expander("See Interpretation"):
         st.info(interpretation)
    # Define the text to be animated in the dialog
    _LOREM_IPSUM = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
        "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    )

    # Define the dialog (using st.dialog) that will show the typewriter effect.
    @st.dialog("AI-Generated Text")
    def show_dialog():
     st.markdown(f"<div>{_LOREM_IPSUM}</div>", unsafe_allow_html=True)

    # When the button is clicked, call the dialog.
    if st.button('Ask AI for Recommendation 💬', use_container_width=True):
        show_dialog()
    return transition_prob, transitions_per_hour, pattern_count
