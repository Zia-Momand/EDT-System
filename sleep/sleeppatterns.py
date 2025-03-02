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
from streamlit_echarts import st_echarts

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
    Files are expected to be named in the format 'YYYY-MM-DD_sleep.json'.
    
    Extracted fields for each day:
      - date: date of sleep (dateOfSleep)
      - minutesAsleep: total sleep duration in minutes
      - deep, light, rem, wake: minutes for each sleep stage from levels.summary
    
    Returns:
      pd.DataFrame: DataFrame containing sleep data for each day.
    """
    data_list = []
    today = datetime.today()
    
    # Loop through today and the previous 6 days
    for i in range(7):
        day = today - timedelta(days=i)
        date_str = day.strftime("%Y-%m-%d")
        file_name = f"{date_str}_sleep.json"
        file_path = os.path.join(directory, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                json_data = json.load(f)
                # Ensure the JSON contains sleep records
                if "sleep" in json_data and json_data["sleep"]:
                    record = json_data["sleep"][0]  # Use the first sleep record
                    date_of_sleep = record.get("dateOfSleep", date_str)
                    minutes_asleep = record.get("minutesAsleep", 0)
                    
                    # Extract sleep stage durations from the levels summary
                    levels = record.get("levels", {})
                    summary = levels.get("summary", {})
                    deep = summary.get("deep", {}).get("minutes", 0)
                    light = summary.get("light", {}).get("minutes", 0)
                    rem   = summary.get("rem", {}).get("minutes", 0)
                    wake  = summary.get("wake", {}).get("minutes", 0)
                    
                    data_list.append({
                        "date": date_of_sleep,
                        "minutesAsleep": minutes_asleep,
                        "deep": deep,
                        "light": light,
                        "rem": rem,
                        "wake": wake
                    })
        else:
            print(f"File not found: {file_path}")
    
    # Sort the data by date (oldest first)
    data_list = sorted(data_list, key=lambda x: x["date"])
    df = pd.DataFrame(data_list)
    return df
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