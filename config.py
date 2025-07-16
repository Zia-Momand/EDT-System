import streamlit as st
# heart_config.py
import os
import json
import time
import requests
import base64
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

import numpy as np
import plotly.express as px
import altair as alt
from logger_manager import LoggerManager

# Fitbit OAuth2 configuration constants
CLIENT_ID = "23PDSY"
CLIENT_SECRET = "54541489aa4c89089e7b20ed97b9087e"
REDIRECT_URI = "https://myfitbitauth.ngrok.io"
AUTHORIZATION_URI = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URI = "https://api.fitbit.com/oauth2/token"

# Define the path for tokens.json (in the same folder as this file)
TOKENS_FILE = os.path.join(os.path.dirname(__file__), "tokens.json")
# Initialize logger
logger = LoggerManager()

# ----- OAuth Flow Functions --------------------------------------------------
#def get_fitbit_auth_url():
    #"""
    #Generate and return the Fitbit OAuth2 URL for user authorization.
    #"""
    #scope = ("heartrate activity oxygen_saturation electrocardiogram "
             #"cardio_fitness sleep respiratory_rate activity temperature")
    #response_type = "code"
    #auth_url = (
        #f"{AUTHORIZATION_URI}"
        #f"?response_type={response_type}"
        #f"&client_id={CLIENT_ID}"
        #f"&redirect_uri={REDIRECT_URI}"
        #f"&scope={scope}"
        #f"&expires_in=604800"  # token expiration in seconds (optional)
    #)
    #return auth_url

# def exchange_code_for_tokens(code):
#     """
#     Exchange the provided authorization code for access and refresh tokens.
#     Adds an 'iat' timestamp to the tokens for expiration checking.
#     """
#     headers = {
#         "Content-Type": "application/x-www-form-urlencoded",
#         "Authorization": "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode(),
#     }
#     data = {
#         "client_id": CLIENT_ID,
#         "grant_type": "authorization_code",
#         "redirect_uri": REDIRECT_URI,
#         "code": code,
#     }

#     response = requests.post(TOKEN_URI, headers=headers, data=data)
#     if response.status_code == 200:
#         tokens = response.json()
#         tokens["iat"] = int(time.time())
#         return tokens
#     else:
#         st.error("Failed to exchange code for tokens.")
#         st.write(response.text)
#         return None

# # ----- Token Initialization --------------------------------------------------
# # Add a checkbox to allow forced reauthorization if needed.
# force_auth = st.checkbox("Force reauthorization (to obtain a fresh token)", value=False)
# if force_auth and os.path.exists(TOKENS_FILE):
#     os.remove(TOKENS_FILE)
#     st.info("Existing token file removed. A new token will be generated.")

# if not os.path.exists(TOKENS_FILE):
#     st.write("No tokens.json file found. Please authorize the application.")
#     st.write("Visit the following URL to authorize and obtain a new authorization code:")
#     #st.write(get_fitbit_auth_url())
#     new_code = st.text_input("Enter the new authorization code:")
#     if new_code:
#         tokens = exchange_code_for_tokens(new_code)
#         if tokens:
#             with open(TOKENS_FILE, "w", encoding="utf-8") as f:
#                 json.dump(tokens, f, indent=4)
#             st.success("New tokens saved successfully!")
#         else:
#             st.error("Failed to exchange new authorization code for tokens.")
#             st.stop()  # Stop execution if tokens are not obtained.
# else:
#     with open(TOKENS_FILE, "r", encoding="utf-8") as f:
#         tokens = json.load(f)

# ----- Token Manager ---------------------------------------------------------
class TokenManager:
    def __init__(self):
        self.tokensFilePath = TOKENS_FILE

    def read_tokens(self):
        """Read tokens from the tokens.json file."""
        if os.path.exists(self.tokensFilePath):
            with open(self.tokensFilePath, "r", encoding="utf-8") as f:
                tokens = json.load(f)
            return tokens
        return None

    def write_tokens(self, tokens):
        """Write updated tokens to tokens.json."""
        with open(self.tokensFilePath, "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=4)

# ----- Token Expiration and Refresh ------------------------------------------
def is_token_expired():
    """
    Check if the current access token is expired.
    Uses the 'iat' (issued-at) timestamp plus the 'expires_in' duration.
    Refresh the token 5 minutes (300 seconds) before actual expiration.
    """
    token_manager = TokenManager()
    tokens = token_manager.read_tokens()
    if not tokens:
        return True
    expires_in = int(tokens.get("expires_in", 0))
    iat = int(tokens.get("iat", 0))
    current_time = int(time.time())
    return current_time >= (iat + expires_in - 300)

def refresh_fitbit_tokens(refresh_token):
    """
    Refresh the access token using the provided refresh token.
    Adds an 'iat' timestamp to the new tokens and saves them.
    """
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
       "grant_type": "refresh_token",
       "refresh_token": refresh_token
    }
    response = requests.post(TOKEN_URI, headers=headers, data=data)
    if response.status_code == 200:
        new_tokens = response.json()
        new_tokens["iat"] = int(time.time())
        # Save new tokens using TokenManager
        token_manager = TokenManager()
        token_manager.write_tokens(new_tokens)
        st.info("Tokens refreshed successfully.")
        return new_tokens
    else:
        st.error("Token refresh failed. Need re-authentication.")
        st.write(response.text)
        return None

# def get_valid_tokens():
#     """
#     Retrieve tokens from storage and refresh if they are expired.
#     Returns valid tokens or stops the app if no valid token is available.
#     """
#     token_manager = TokenManager()
#     tokens = token_manager.read_tokens()
#     if not tokens:
#         st.error("No tokens available. Please authenticate.")
#         st.stop()
#     if is_token_expired():
#         st.info("Access token expired. Attempting to refresh...")
#         refresh_token_val = tokens.get("refresh_token")
#         new_tokens = refresh_fitbit_tokens(refresh_token_val)
#         if new_tokens:
#             tokens = new_tokens
#         else:
#             st.error("Token refresh failed. Please reauthenticate.")
#             st.stop()
#     return tokens

# ----- Fitbit API URL Helpers -----------------------------------------------
# def get_fitbit_api_url_for_day(date_str, detail_level="1min"):
#     """
#     Return the intraday API URL for a given day.
#     Date should be a string in YYYY-MM-DD format.
#     """
#     tokens = get_valid_tokens()
#     USER_ID = tokens.get("user_id")
#     return f"https://api.fitbit.com/1/user/{USER_ID}/activities/heart/date/{date_str}/1d/{detail_level}.json"

# def get_fitbit_api_url_for_range(start_date, end_date):
#     """
#     Return the API URL for a date range.
#     Dates should be strings in YYYY-MM-DD format.
#     (Note: The date-range endpoint returns summary data per day.)
#     """
#     tokens = get_valid_tokens()
#     USER_ID = tokens.get("user_id")
#     return f"https://api.fitbit.com/1/user/{USER_ID}/activities/heart/date/{start_date}/{end_date}.json"

# ----- Data Fetch and Load Functions ----------------------------------------
# def update_heart_rate_data_for_day(date_str, label, detail_level="1min"):
#     """
#     Download heart rate data (in JSON format) for the specified date using the intraday endpoint.
#     Saves the file in static/data as '<date>_<label>.json' and returns the JSON data.
#     Automatically refreshes the token if needed.
#     """
#     api_url = get_fitbit_api_url_for_day(date_str, detail_level)
#     tokens = get_valid_tokens()
#     access_token = tokens.get("access_token")
#     headers = {"Authorization": f"Bearer {access_token}"}
#     response = requests.get(api_url, headers=headers)
#     if response.status_code == 200:
#         json_data = response.json()
#     elif response.status_code == 401:
#         # Token might be expired; try refreshing and retrying
#         tokens = refresh_fitbit_tokens(tokens.get("refresh_token"))
#         if tokens:
#             access_token = tokens.get("access_token")
#             headers = {"Authorization": f"Bearer {access_token}"}
#             response = requests.get(api_url, headers=headers)
#             if response.status_code == 200:
#                 json_data = response.json()
#             else:
#                 st.error(f"Error after token refresh: {response.status_code}")
#                 st.write(response.text)
#                 return None
#         else:
#             st.error("Token refresh failed. Please reauthenticate.")
#             return None
#     else:
#         st.error(f"Error fetching heart rate data: {response.status_code}")
#         st.write(response.text)
#         return None

#     data_dir = os.path.join("static", "data")
#     os.makedirs(data_dir, exist_ok=True)
#     file_name = f"{date_str}_{label}.json"
#     file_path = os.path.join(data_dir, file_name)
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(json_data, f, indent=2)
#     return json_data
@logger.log_user_activity(task_name="Fetch Heart Rate Data", is_final=True)
def load_heart_rate_data_for_day(date_str, label):
    """
    Load the heart rate JSON file for the given date (and label) from static/data,
    and return the 'dataset' (intraday measurements).
    """
    file_path = os.path.join("static", "data", "2025-05-02_24h.json")
    if not os.path.exists(file_path):
        st.warning(f"No heart rate data file found for {date_str} ({label}). Please refresh.")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data.get("activities-heart-intraday", {}).get("dataset", [])

# def update_heart_rate_data_for_range(start_date, end_date):
#     """
#     Download heart rate data for the specified date range using the range endpoint.
#     Saves the file in static/data as '<start_date>_to_{end_date}_week.json'
#     and returns the JSON data.
#     Automatically refreshes the token if needed.
#     """
#     api_url = get_fitbit_api_url_for_range(start_date, end_date)
#     tokens = get_valid_tokens()
#     access_token = tokens.get("access_token")
#     headers = {"Authorization": f"Bearer {access_token}"}
#     response = requests.get(api_url, headers=headers)
#     if response.status_code == 200:
#         json_data = response.json()
#     elif response.status_code == 401:
#         tokens = refresh_fitbit_tokens(tokens.get("refresh_token"))
#         if tokens:
#             access_token = tokens.get("access_token")
#             headers = {"Authorization": f"Bearer {access_token}"}
#             response = requests.get(api_url, headers=headers)
#             if response.status_code == 200:
#                 json_data = response.json()
#             else:
#                 st.error(f"Error after token refresh: {response.status_code}")
#                 st.write(response.text)
#                 return None
#         else:
#             st.error("Token refresh failed. Please reauthenticate.")
#             return None
#     else:
#         st.error(f"Error fetching heart rate data: {response.status_code}")
#         st.write(response.text)
#         return None

#     data_dir = os.path.join("static", "data")
#     os.makedirs(data_dir, exist_ok=True)
#     file_name = f"{start_date}_to_{end_date}_week.json"
#     file_path = os.path.join(data_dir, file_name)
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(json_data, f, indent=2)
#     return json_data
#@logger.log_user_activity(task_name="Healthcare Recommendation", is_final=True)
def load_heart_rate_data_for_range(start_date, end_date):
    """
    Load the heart rate JSON file for the given date range from static/data,
    and return the dataset.
    (For range data, Fitbit returns summary data per day in the 'activities-heart' key.)
    """
    file_path = os.path.join("static", "data", f"{start_date}_to_{end_date}_week.json")
    if not os.path.exists(file_path):
        st.warning("No heart rate data file found for the week. Please refresh.")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data.get("activities-heart", [])
@logger.log_user_activity(task_name="Fetech Weekly Heart Rate Data", is_final=True)
def load_weekly_heart_rate_data():
    """
    Load the most recent 7 days of heart rate JSON files from static/data,
    extracting weekly trends from available '<date>_24h.json' files.
    """
    base_path = os.path.join("static", "data")
    today = date.today()
    weekly_data = {}

    for i in range(7):
        date_str = '2025-02-27'#(today - timedelta(days=i)).strftime("%Y-%m-%d")
        file_path = os.path.join(base_path, f"{date_str}_24h.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                daily_data = json_data.get("activities-heart-intraday", {}).get("dataset", [])
                if daily_data:
                    weekly_data[date_str] = daily_data

    if not weekly_data:
        st.warning("No heart rate data available for the past week.")
        return None

    return weekly_data

# ----- Data Processing and Plotting Functions -------------------------------
@logger.log_user_activity(task_name="Process Heart Rate Data", is_final=True)
def process_heart_rate_data(data):
    """Process raw heart rate data into a DataFrame (for intraday data)."""
    if not data:
        return None
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

@logger.log_user_activity(task_name="Calculate Heart Rate Variability", is_final=True)
def calculate_rmssd_for_5min_intervals(data):
    """
    Calculate RMSSD for each 5-minute interval and visualize with color-coded thresholds.
    """
    if not data or len(data) < 5:
        st.warning("Insufficient data to calculate RMSSD.")
        return None

    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    rmssd_results = []
    time_labels = []

    for i in range(0, len(df) - 5, 5):
        window = df.iloc[i : i + 5]
        time_diff = np.diff(window["Time"].astype(np.int64) // 10**9) / 60
        if not np.all(time_diff == 1):
            continue

        rr_intervals = 60000 / window["value"]
        rr_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(rr_diffs**2))
        rmssd_results.append(rmssd)
        time_labels.append(window.iloc[0]["Time"])  # 🟢 Keep datetime

    if not rmssd_results:
        st.warning("No valid RMSSD results due to inconsistent time intervals.")
        return None

    df_plot = pd.DataFrame({
        "Time": time_labels,
        "RMSSD": rmssd_results
    })

    # Base chart
    line = alt.Chart(df_plot).mark_line(color='blue').encode(
        x=alt.X('Time:T', title='Time'),
        y=alt.Y('RMSSD:Q', title='RMSSD (ms)')
    )

    # Threshold lines
    normal = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(color='green', strokeDash=[2, 2]).encode(y='y')
    moderate = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='orange', strokeDash=[4, 2]).encode(y='y')
    low = alt.Chart(pd.DataFrame({'y': [10]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y')

    st.altair_chart(line + normal + moderate + low, use_container_width=True)

    return time_labels, rmssd_results


# def fetch_sleep_data():
#     """
#     Fetch sleep data from Fitbit API for today's date and save it as a JSON file.
#     Uses get_valid_tokens() to retrieve stored Fitbit tokens.
#     """
#     current_date = datetime.today().strftime("%Y-%m-%d")
#     tokens = get_valid_tokens()

#     USER_ID = tokens.get("user_id")
#     access_token = tokens.get("access_token")
#     if not USER_ID or not access_token:
#         st.error("Missing user ID or access token in stored tokens.")
#         return None

#     FITBIT_SLEEP_API = f"https://api.fitbit.com/1.2/user/{USER_ID}/sleep/date/{current_date}.json"
#     headers = {"Authorization": f"Bearer {access_token}"}
#     response = requests.get(FITBIT_SLEEP_API, headers=headers)

#     if response.status_code == 200:
#         sleep_data = response.json()
#         os.makedirs(os.path.join("static", "data", "sleep"), exist_ok=True)
#         file_path = os.path.join("static", "data", "sleep", f"{current_date}_sleep.json")
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(sleep_data, f, indent=4)
#         return sleep_data
#     else:
#         st.error(f"Failed to fetch sleep data: {response.status_code} - {response.text}")
#         return None
#### =============================== Sleep stages process functions =============================
@logger.log_user_activity(task_name="Fetch Sleep Data", is_final=True)
def load_sleep_data(date_str=None):

    base_dir = os.path.join("static", "data", "sleep")
    
    if date_str is None:
        date_str = datetime.today().strftime("%Y-%m-%d")
    
    target_file = f"{date_str}_sleep.json"
    file_path = os.path.join(base_dir, target_file)

    if not os.path.exists(file_path):
        # Fallback: find the latest available sleep file
        def extract_date(filename):
            try:
                return datetime.strptime(filename.split("_")[0], "%Y-%m-%d")
            except ValueError:
                return None

        sleep_files = [
            f for f in os.listdir(base_dir)
            if f.endswith("_sleep.json") and extract_date(f) is not None
        ]

        if not sleep_files:
            return None

        # Get the latest file by date
        latest_file = sorted(sleep_files, key=extract_date, reverse=True)[0]
        file_path = os.path.join(base_dir, latest_file)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {file_path}. Reason: {e}")
        return None
@logger.log_user_activity(task_name="Process Sleep Data", is_final=True)
def process_sleep_data(sleep_data):
    """
    Process Fitbit sleep data into a structured DataFrame.
    Extracts sleep stages from the JSON, converts timestamps, computes
    end times, and creates a label combining the sleep level with its duration.
    Also adds formatted time strings for tooltip display.
    """
    if not sleep_data or "sleep" not in sleep_data or len(sleep_data["sleep"]) == 0:
        st.warning("No sleep data available.")
        return None

    # Extract sleep stages from the first sleep record
    sleep_stages = sleep_data["sleep"][0]["levels"]["data"]
    df = pd.DataFrame(sleep_stages)
    
    # Convert starting time and compute end time for each stage
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    df["endTime"] = df["dateTime"] + pd.to_timedelta(df["seconds"], unit="s")
    
    # Create string columns that show only the time for tooltips
    df["start_time"] = df["dateTime"].dt.strftime("%H:%M:%S")
    df["end_time"] = df["endTime"].dt.strftime("%H:%M:%S")
    
    # Format duration as "X min Y sec"
    df["duration_text"] = df["seconds"].apply(lambda s: f"{s//60} min {s % 60} sec")
    
    # Create a label combining sleep level and its duration
    df["label"] = df.apply(lambda row: f"{row['level'].capitalize()} ({row['duration_text']})", axis=1)
    return df
@logger.log_user_activity(task_name="Plot Sleep Data", is_final=True)
def plot_sleep_stages(df):
    """
    Create a clear sleep stages timeline using Altair.
    - Each sleep stage is shown as a horizontal colored bar.
    - A dotted line connects the end of one stage to the start of the next.
    - The tooltip shows only start/end times (HH:MM:SS).
    """
    if df is None or df.empty:
        st.warning("No valid sleep data to plot.")
        return

    # Sort data by start time
    df = df.copy().sort_values("dateTime").reset_index(drop=True)
    
    # Create string columns showing only the time
    df["start_time_str"] = df["dateTime"].dt.strftime("%H:%M:%S")
    df["end_time_str"]   = df["endTime"].dt.strftime("%H:%M:%S")
    
    # Define color mapping for sleep levels
    sleep_colors = {
        "wake": "#c19a6b",
        "rem": "#b19cd9",
        "light": "#654ea3",
        "deep": "#000033",
    }

    # 1) BARS: horizontal bars from dateTime to endTime
    base = alt.Chart(df).encode(
        y=alt.Y('level:N', title='Sleep Stage',
                sort=["wake", "rem", "light", "deep"])
    )

    bars = base.mark_bar().encode(
        x=alt.X('dateTime:T', title='Time'),
        x2=alt.X2('endTime:T'),
        color=alt.Color('level:N',
                        scale=alt.Scale(domain=list(sleep_colors.keys()),
                                        range=list(sleep_colors.values())),
                        legend=None),
        tooltip=[
            alt.Tooltip("level:N", title="Sleep Stage"),
            alt.Tooltip("start_time_str:N", title="Start Time"),
            alt.Tooltip("end_time_str:N", title="End Time"),
        ]
    )

    # 2) TRANSITIONS: dotted lines between consecutive stages
    #    Build a small dataframe that pairs (endTime_i, level_i) -> (dateTime_{i+1}, level_{i+1})
    lines_list = []
    line_id = 0

    for i in range(len(df) - 1):
        current_end = df.loc[i, 'endTime']
        current_stage = df.loc[i, 'level']
        next_start = df.loc[i+1, 'dateTime']
        next_stage = df.loc[i+1, 'level']

        # If you only want a line when there's an actual time gap, you can check:
        # if next_start > current_end:
        #     ...

        lines_list.append({
            'line_id': line_id,
            'time': current_end,
            'stage': current_stage
        })
        lines_list.append({
            'line_id': line_id,
            'time': next_start,
            'stage': next_stage
        })
        line_id += 1

    df_lines = pd.DataFrame(lines_list)

    transitions = alt.Chart(df_lines).mark_line(
        strokeDash=[2, 2],       # Dotted line
        strokeWidth=2,          # Thicker stroke for visibility
        interpolate='step-after' # Creates a step-like zigzag
    ).encode(
        x='time:T',
        y=alt.Y('stage:N', sort=["wake", "rem", "light", "deep"]),
        color=alt.Color('stage:N',
                        scale=alt.Scale(domain=list(sleep_colors.keys()),
                                        range=list(sleep_colors.values())),
                        legend=None),
        detail='line_id:N'
    )

    # 3) COMBINE bars + transitions (transitions drawn last -> on top)
    chart = (bars + transitions).properties(
        title="Sleep Stages Timeline",
        width=600,
        height=300
    )

    st.altair_chart(chart, use_container_width=True)

###-----------   Plotting sleep stages trends --------------------
@logger.log_user_activity(task_name="Show Sleep Trends", is_final=True)
def plot_sleep_trends(start_date, end_date, period='daily'):

    date_range = pd.date_range(start=start_date, end=end_date)
    records = []  # To collect aggregated daily data
    
    for dt in date_range:
        date_str = dt.strftime("%Y-%m-%d")
        sleep_json = load_sleep_data(date_str)
        if sleep_json is None:
            continue  # Skip if no file exists
        df_sleep = process_sleep_data(sleep_json)
        if df_sleep is None or df_sleep.empty:
            continue
        # Group by sleep stage and sum seconds for that day
        grouped = df_sleep.groupby("level")["seconds"].sum().reset_index()
        for _, row in grouped.iterrows():
            records.append({
                "date": dt,
                "level": row["level"],
                "seconds": row["seconds"]
            })
    
    if not records:
        # Return an empty chart if no data is found
        return alt.Chart(pd.DataFrame()).mark_line().encode()
    
    df_trends = pd.DataFrame(records)
    # Map internal levels to friendly labels
    level_mapping = {"wake": "Awake", "rem": "REM", "light": "Light", "deep": "Deep"}
    df_trends["stage"] = df_trends["level"].map(level_mapping)
    df_trends["minutes"] = df_trends["seconds"] / 60.0
    
    # If weekly aggregation is requested, sum data by week
    if period == 'weekly':
        df_trends.set_index("date", inplace=True)
        weekly = df_trends.groupby("stage").resample("W")["minutes"].sum().reset_index()
        df_plot = weekly
    else:
        df_plot = df_trends.copy()
    
    chart = alt.Chart(df_plot).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("minutes:Q", title="Total Minutes"),
        color=alt.Color("stage:N", title="Sleep Stage",
                        scale=alt.Scale(
                            domain=["Awake", "REM", "Light", "Deep"],
                            range=["#c19a6b", "#b19cd9", "#654ea3", "#000033"]
                        )),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("stage:N", title="Sleep Stage"),
            alt.Tooltip("minutes:Q", title="Total Minutes", format=".1f")
        ]
    ).properties(
        title=f"Sleep Stage Trends ({'Weekly' if period=='weekly' else 'Daily'})",
        width=600,
        height=400
    )
    
    return chart
@logger.log_user_activity(task_name="Show Daily Heart Rate", is_final=True)
def plot_heart_rate_for_day(date_str, label):
    """
    Plot heart rate data for a given day (intraday data) with color-coded threshold indicators.
    """
    dataset = load_heart_rate_data_for_day(date_str, label)
    if not dataset:
        st.warning("No heart rate data available.")
        return

    df = process_heart_rate_data(dataset)
    if df is None or df.empty:
        st.warning("No valid heart rate data to plot.")
        return

    df_plot = df.reset_index().rename(columns={"time": "Time", "value": "Heart Rate"})

    # Base heart rate line chart
    line = alt.Chart(df_plot).mark_line(color='blue').encode(
        x='Time:T',
        y='Heart Rate:Q'
    ).properties(
        title='Heart Rate Over Time'
    )

    # Threshold indicators
    normal_line = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(color='green', strokeDash=[2, 2]).encode(y='y')
    elevated_line = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(color='orange', strokeDash=[4, 2]).encode(y='y')
    abnormal_line = alt.Chart(pd.DataFrame({'y': [130]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y')

    # Combine all
    chart = line + normal_line + elevated_line + abnormal_line

    st.altair_chart(chart, use_container_width=True)

@logger.log_user_activity(task_name="Plot Weekly Heart Rate Trends", is_final=True)
def plot_weekly_heart_rate():
    """
    Plot weekly heart rate trends using Streamlit's line_chart.
    """
    weekly_data = load_weekly_heart_rate_data()
    if not weekly_data:
        st.warning("No heart rate data available for the past week.")
        return

    data_list = []
    for day, records in weekly_data.items():
        for record in records:
            data_list.append({"Time": f"{day} {record['time']}", "Heart Rate": record["value"]})

    df = pd.DataFrame(data_list)
    if df.empty:
        st.warning("No valid heart rate data found for plotting.")
        return
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")
    st.line_chart(df.set_index("Time")["Heart Rate"])

@logger.log_user_activity(task_name="Plot Heart Rate Variability", is_final=True)
def plot_rmssd_trends(date_str, label):
    """
    Load heart rate data, calculate RMSSD for 5-minute intervals, and plot the trends.
    """
    heart_rate_data = load_heart_rate_data_for_day(date_str, label)
    if not heart_rate_data:
        return
    result = calculate_rmssd_for_5min_intervals(heart_rate_data)
    if not result:
        return
    #time_labels, rmssd_values = result
    #df_plot = pd.DataFrame({"Time": time_labels, "RMSSD": rmssd_values})
    #st.line_chart(df_plot.set_index("Time")["RMSSD"])


    #================== Get SpO2 data ============================

# def fetch_spo2_data():
#     """
#     Fetch SPO2 data from the Fitbit API for today's date and save it as a JSON file.
#     Uses get_valid_tokens() to retrieve stored Fitbit tokens.
    
#     The JSON file is saved in the folder static/data/spo2 as {current_date}_spo2.json.
    
#     Returns:
#       dict: The SPO2 data retrieved from the Fitbit API, or None if an error occurs.
#     """
#     #current_date = datetime.today().strftime("%Y-%m-%d")
#     yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
#     tokens = get_valid_tokens()
#     #current_date ="2025-03-09"

#     access_token = tokens.get("access_token")
#     if not access_token:
#         st.error("Missing access token in stored tokens.")
#         return None

#     # Construct the correct Fitbit SPO2 API endpoint URL
#     FITBIT_SPO2_API = f"https://api.fitbit.com/1/user/-/spo2/date/{yesterday}/all.json"
#     headers = {"Authorization": f"Bearer {access_token}"}
#     response = requests.get(FITBIT_SPO2_API, headers=headers)

#     if response.status_code == 200:
#         spo2_data = response.json()
#         # Create the target folder if it doesn't exist
#         folder = os.path.join("static", "data", "spo2")
#         os.makedirs(folder, exist_ok=True)
#         file_path = os.path.join(folder, f"{yesterday}_spo2.json")
#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(spo2_data, f, indent=4)
#         return spo2_data
#     else:
#         st.error(f"Failed to fetch SPO2 data: {response.status_code} - {response.text}")
#         return None