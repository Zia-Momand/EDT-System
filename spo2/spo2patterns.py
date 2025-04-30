import os
import json
from datetime import datetime, timedelta
import pandas as pd
import altair as alt
import streamlit as st
import requests 
from streamlit_echarts import st_echarts
class SPO2Analyzer:
    def __init__(self):
        pass
    @staticmethod
    def plot_spo2_last_night():
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        file_path = os.path.join("static", "data", "spo2", f"{yesterday}_spo2.json")
        
        if not os.path.exists(file_path):
            st.warning(f"No SPO2 data found for {yesterday}.")
            return
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # New data format: use "dateTime" and "minutes" keys.
        df = pd.DataFrame(data["minutes"])
        # Convert the "minute" field to a datetime column.
        df["time_dt"] = pd.to_datetime(df["minute"])
        
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("time_dt:T", title="Time"),
            y=alt.Y("value:Q", title="SPO2 Level (%)", scale=alt.Scale(domain=[80, 100])),
            tooltip=[
                alt.Tooltip("time_dt:T", title="Time", format="%H:%M"),
                alt.Tooltip("value:Q", title="SPO2")
            ]
        ).properties(
            title=f"SPO2 Levels for Last Night ({data['dateTime']})",
            width=600,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)
    @staticmethod
    def plot_spo2_weekly():
        end_date = datetime.today() - timedelta(days=1)  # assume last full day
        start_date = end_date - timedelta(days=6)
        date_range = pd.date_range(start=start_date, end=end_date)
        all_records = []
        
        for dt in date_range:
            date_str = dt.strftime("%Y-%m-%d")
            file_path = os.path.join("static", "data", "spo2", f"{date_str}_spo2.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rec in data["spo2"]:
                    rec["date"] = data["date"]
                    # Create a combined datetime column.
                    rec["time_dt"] = pd.to_datetime(data["date"] + " " + rec["time"])
                all_records.extend(data["spo2"])
        
        if not all_records:
            st.warning("No SPO2 data available for the last week.")
            return
        
        df_week = pd.DataFrame(all_records)
        
        chart = alt.Chart(df_week).mark_line(point=True).encode(
            x=alt.X("time_dt:T", title="Time"),
            y=alt.Y("value:Q", title="SPO2 Level (%)"),
            color=alt.Color("date:N", title="Date"),
            tooltip=[
                alt.Tooltip("time_dt:T", title="Time", format="%H:%M"),
                alt.Tooltip("value:Q", title="SPO2"),
                alt.Tooltip("date:N", title="Date")
            ]
        ).properties(
            title="SPO2 Levels Over the Last Week",
            width=600,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)

# ------------------------------
#======== MAX, Min , and Avg SPO2 levels =====================
    @staticmethod
    def plot_spo2_liquid_fill(date_str=None):
        
        if date_str is None:
            date_str = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            
        file_path = os.path.join("static", "data", "spo2", f"{date_str}_spo2.json")
        if not os.path.exists(file_path):
            st.warning(f"No SPO2 data found for {date_str}.")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create a DataFrame from the 'minutes' list
        df = pd.DataFrame(data["minutes"])
        #st.write(df.head())
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        if df.empty:
            st.warning("No SPO2 values available.")
            return

        # Compute min, average, and max SPO2 values
        min_val = df["value"].min()
        avg_val = df["value"].mean()
        max_val = df["value"].max()
        
        # Convert to fractions (liquid fill expects [0, 1])
        min_frac = min_val / 100.0
        avg_frac = avg_val / 100.0
        max_frac = max_val / 100.0
        # Helper function to build a single-wave liquid fill chart
        def build_liquid_option(percentage, label_text):
            return {
                "series": [
                    {
                        "type": "liquidFill",
                        "data": [percentage],
                        "radius": "80%",
                        "center": ["50%", "50%"],  # ensures the chart is centered
                        "backgroundStyle": {
                            "borderWidth": 5,
                            "borderColor": "#1565C0",
                            "color": "transparent"
                        },
                        "outline": {
                            "show": True,
                            "borderDistance": 0,
                            "itemStyle": {
                                "borderWidth": 5,
                                "borderColor": "#1565C0"
                            }
                        },
                        "color": ["#1976d2"],
                        "label": {
                            "formatter": f"{label_text:.1f}%",
                            "fontSize": 20,
                            "color": "#fff"
                        },
                        "waveAnimation": True,
                        "waveNum": 1,
                        "amplitude": 5
                    }
                ]
            }

        # Build three separate chart configs for min, avg, and max
        option_min = build_liquid_option(min_frac, min_val)
        option_avg = build_liquid_option(avg_frac, avg_val)
        option_max = build_liquid_option(max_frac, max_val)

        # Show the three charts side by side
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Minimum SpO2**")
            st_echarts(options=option_min, height="200px")
        with cols[1]:
            st.markdown("**Average SpO2**")
            st_echarts(options=option_avg, height="200px")
        with cols[2]:
            st.markdown("**Maximum SpO2**")
            st_echarts(options=option_max, height="200px")

        # ---- Trigger Alert Based on min_val ----
        if min_val < 67:
            message = "Cyanosis develops when SpO₂ level drops below 67%."
            st.error(message)
        elif min_val < 80:
            message = "The brain can be affected when SpO₂ level falls below 80–85%."
            st.error(message)
        elif min_val < 95:
            message = "SpO₂ below 95% can be a sign of hypoxemia; consider seeing a healthcare provider."
            st.warning(message)
        else:
            message = "Normal oxygen levels (95%–100%)."
            st.success(message)

        # ----- Trigger AI recommendation if SpO₂ is abnormal -----
        if min_val < 95:
            if st.button('Get AI Recommendation for Better Care 💬', use_container_width=True):
                abnormal_text = f"Minimum SpO₂ recorded: {min_val:.1f}%. {message}"
                with st.spinner("Analyzing abnormality and generating recommendation..."):
                    advice = SPO2Analyzer.get_ai_recommendation(abnormal_text)
                st.markdown("### 🧠 AI-Generated Care Recommendations")
                st.markdown(advice)
    #-------- get AI recommendations function
    @staticmethod
    def get_ai_recommendation(abnormal_details: str) -> str:
            """
            Generate structured AI recommendations for SpO₂ abnormalities using bullet-point formatting.
            """

            # Define a structured prompt tailored for SpO₂ context
            prompt_template = """A caregiver is monitoring a patient’s blood oxygen saturation (SpO₂), and the following abnormalities were detected:

        {abnormal_details}

        Based on clinical norms and safety thresholds, provide **concise and meaningful recommendations for caregivers**.
        Ensure **a maximum of 4 bullet points**, each addressing a specific concern with practical advice related to oxygen levels.

        ### **Format your response EXACTLY like this:**

        - **[First abnormality]** → [Short recommendation]  

        - **[Second abnormality]** → [Short recommendation]  

        - **[Third abnormality]** → [Short recommendation]  

        - **[Fourth abnormality]** → [Short recommendation]  

        🚨 **DO NOT** include any introduction, summary, or extra text. **ONLY provide bullet points in the format above.**
        """

            from langchain.prompts import PromptTemplate
            from langchain.chat_models import ChatOpenAI
            from langchain.chains import LLMChain
            import re

            # Build the prompt
            prompt = PromptTemplate(template=prompt_template, input_variables=["abnormal_details"])

            try:
                # Initialize LLM and chain
                llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)
                chain = LLMChain(llm=llm, prompt=prompt)

                # Run the LLM with the abnormal SpO2 description
                recommendation = chain.run(abnormal_details=abnormal_details).strip()

                # Format correction: ensure proper line breaks before bullets
                recommendation = re.sub(r"\n\s*-\s\*\*", "\n\n- **", recommendation)

                return recommendation

            except Exception as e:
                return f"Error generating recommendation: {e}"

    @staticmethod
    def plot_hr_spo2_last_night():
        import re

        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        spo2_path = os.path.join("static", "data", "spo2", f"{yesterday}_spo2.json")
        hr_path = os.path.join("static", "data", f"{yesterday}_24h.json")

        # --- Load SpO2 Data ---
        if not os.path.exists(spo2_path):
            st.warning(f"No SPO2 data found for {yesterday}.")
            return
        
        with open(spo2_path, "r", encoding="utf-8") as f:
            spo2_data = json.load(f)
        spo2_df = pd.DataFrame(spo2_data["minutes"])
        spo2_df["time_dt"] = pd.to_datetime(spo2_df["minute"])
        spo2_df.rename(columns={"value": "SpO2"}, inplace=True)

        # --- Load Heart Rate Data ---
        if not os.path.exists(hr_path):
            st.warning(f"No Heart Rate data found for {yesterday}.")
            return

        with open(hr_path, "r", encoding="utf-8") as f:
            hr_data = json.load(f)

        hr_df = pd.DataFrame(hr_data["activities-heart-intraday"]["dataset"])
        if hr_df.empty:
            st.warning("Heart rate dataset is empty.")
            return

        hr_df["time_dt"] = pd.to_datetime(f"{yesterday} " + hr_df["time"])
        hr_df.rename(columns={"value": "HR"}, inplace=True)

        # --- Align SpO2 and HR by nearest timestamps (within 1 minute) ---
        aligned_df = pd.merge_asof(
            spo2_df.sort_values("time_dt"),
            hr_df.sort_values("time_dt"),
            on="time_dt",
            direction="nearest",
            tolerance=pd.Timedelta("60s")
        )

        # Drop rows with no HR match
        aligned_df.dropna(subset=["HR"], inplace=True)

        # --- Plot both HR and SpO2 ---
        spo2_line = alt.Chart(aligned_df).mark_line(color="#1976d2").encode(
            x=alt.X("time_dt:T", title="Time"),
            y=alt.Y("SpO2:Q", title="SpO₂ (%)", scale=alt.Scale(domain=[80, 100])),
            tooltip=[alt.Tooltip("time_dt:T", title="Time", format="%H:%M"),
                    alt.Tooltip("SpO2:Q", title="SpO₂")],
        )

        hr_line = alt.Chart(aligned_df).mark_line(color="#f44336").encode(
            x=alt.X("time_dt:T"),
            y=alt.Y("HR:Q", title="Heart Rate (BPM)", scale=alt.Scale(domain=[50, 130])),
            tooltip=[alt.Tooltip("HR:Q", title="Heart Rate (BPM)")],
        )

        chart = alt.layer(spo2_line, hr_line).resolve_scale(y='independent').properties(
            title=f"SPO₂ and Heart Rate Trends on {yesterday}",
            width=700,
            height=350
        )

        st.altair_chart(chart, use_container_width=True)

        





