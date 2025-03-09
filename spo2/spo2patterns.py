import os
import json
from datetime import datetime, timedelta
import pandas as pd
import altair as alt
import streamlit as st
import requests 

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
            y=alt.Y("value:Q", title="SPO2 Level (%)"),
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
        """
        Loads SPO2 JSON files for the last 7 days from static/data/spo2,
        aggregates the data, and visualizes the SPO2 levels for each night.
        Each night is represented as a separate line, colored by date.
        """
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

