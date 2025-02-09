# heart_config.py
import os
import json
import requests
import base64
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# Fitbit OAuth2 configuration constants
CLIENT_ID = "23PDSY"
CLIENT_SECRET = "54541489aa4c89089e7b20ed97b9087e"
REDIRECT_URI = "https://elderlycare.loca.lt/callback"
AUTHORIZATION_URI = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URI = "https://api.fitbit.com/oauth2/token"

# ----- Token Manager ---------------------------------------------------------
class TokenManager:
    def __init__(self):
        self.tokensFilePath = os.path.join(os.path.dirname(__file__), "tokens.json")

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
            json.dump(tokens, f)

# ----- OAuth Flow Functions --------------------------------------------------
def get_fitbit_auth_url():
    """
    Returns the URL where the user should be redirected to authorize the application.
    Adjust the scope as needed.
    """
    scope = "heartrate activity"  # adjust scopes based on your needs
    response_type = "code"
    auth_url = (
        f"{AUTHORIZATION_URI}"
        f"?response_type={response_type}"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope={scope}"
        f"&expires_in=604800"  # optional: token expiration in seconds
    )
    return auth_url

def exchange_code_for_tokens(code):
    """
    After the user authorizes the app and is redirected to your REDIRECT_URI,
    exchange the provided authorization code for access and refresh tokens.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    # Create a basic auth header using client_id and client_secret.
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers["Authorization"] = f"Basic {auth_header}"

    data = {
        "client_id": CLIENT_ID,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }

    response = requests.post(TOKEN_URI, headers=headers, data=data)
    if response.status_code == 200:
        tokens = response.json()
        return tokens
    else:
        st.error("Failed to exchange code for tokens.")
        st.write(response.text)
        return None

def refresh_fitbit_tokens(refresh_token):
    """
    Refresh the access token using the stored refresh token.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers["Authorization"] = f"Basic {auth_header}"

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID
    }

    response = requests.post(TOKEN_URI, headers=headers, data=data)
    if response.status_code == 200:
        new_tokens = response.json()
        token_manager = TokenManager()
        token_manager.write_tokens(new_tokens)
        return new_tokens
    else:
        st.error(f"Error refreshing token: {response.status_code}")
        st.write(response.text)
        return None

# ----- Helper Functions for Building API URLs -----------------------------
def get_fitbit_api_url_for_day(date, detail_level="1min"):
    """
    Return the intraday API URL for a given day.
    Date should be a string in YYYY-MM-DD format.
    """
    token_manager = TokenManager()
    tokens = token_manager.read_tokens()
    if not tokens:
        st.error("No Fitbit tokens found. Please authenticate and store tokens.")
        st.stop()
    USER_ID = tokens.get("user_id")
    return f"https://api.fitbit.com/1/user/{USER_ID}/activities/heart/date/{date}/1d/{detail_level}.json"

def get_fitbit_api_url_for_range(start_date, end_date):
    """
    Return the API URL for a date range.
    Dates should be strings in YYYY-MM-DD format.
    (Note: The date-range endpoint returns summary data per day.)
    """
    token_manager = TokenManager()
    tokens = token_manager.read_tokens()
    if not tokens:
        st.error("No Fitbit tokens found. Please authenticate and store tokens.")
        st.stop()
    USER_ID = tokens.get("user_id")
    return f"https://api.fitbit.com/1/user/{USER_ID}/activities/heart/date/{start_date}/{end_date}.json"

# ----- Update & Load Functions for Daily Data ------------------------------
def update_heart_rate_data_for_day(date, label, detail_level="1min"):
    """
    Download heart rate data (in JSON format) for the specified date using the intraday endpoint.
    Save the file in static/data as '<date>_<label>.json' and return the JSON data.
    """
    api_url = get_fitbit_api_url_for_day(date, detail_level)
    token_manager = TokenManager()
    tokens = token_manager.read_tokens()
    if not tokens:
        st.error("No Fitbit tokens found. Please authenticate.")
        st.stop()
    FITBIT_ACCESS_TOKEN = tokens.get("access_token")
    
    headers = {"Authorization": f"Bearer {FITBIT_ACCESS_TOKEN}"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        json_data = response.json()
        data_dir = os.path.join("static", "data")
        os.makedirs(data_dir, exist_ok=True)
        file_name = f"{date}_{label}.json"
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        return json_data
    elif response.status_code == 401:
        st.error("401 Unauthorized. The access token may be expired.")
        return None
    else:
        st.error(f"Error fetching heart rate data: {response.status_code}")
        return None

def load_heart_rate_data_for_day(date, label):
    """
    Load the heart rate JSON file for the given date (and label) from static/data,
    and return the 'dataset' (intraday measurements).
    """
    file_path = os.path.join("static", "data", f"{date}_{label}.json")
    if not os.path.exists(file_path):
        st.warning(f"No heart rate data file found for {date} ({label}). Please refresh.")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data.get("activities-heart-intraday", {}).get("dataset", [])

# ----- Update & Load Functions for Range Data (Week Trends) ----------------
def update_heart_rate_data_for_range(start_date, end_date):
    """
    Download heart rate data for the specified date range using the range endpoint.
    Save the file in static/data as '<start_date>_to_{end_date}_week.json'
    and return the JSON data.
    """
    api_url = get_fitbit_api_url_for_range(start_date, end_date)
    token_manager = TokenManager()
    tokens = token_manager.read_tokens()
    if not tokens:
        st.error("No Fitbit tokens found. Please authenticate.")
        st.stop()
    FITBIT_ACCESS_TOKEN = tokens.get("access_token")
    
    headers = {"Authorization": f"Bearer {FITBIT_ACCESS_TOKEN}"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        json_data = response.json()
        data_dir = os.path.join("static", "data")
        os.makedirs(data_dir, exist_ok=True)
        file_name = f"{start_date}_to_{end_date}_week.json"
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        return json_data
    elif response.status_code == 401:
        st.error("401 Unauthorized. The access token may be expired.")
        return None
    else:
        st.error(f"Error fetching heart rate data: {response.status_code}")
        return None

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

# ----- Processing and Plotting Functions ------------------------------------
def process_heart_rate_data(data):
    """Process raw heart rate data into a DataFrame (for intraday data)."""
    if not data:
        return None
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])  # Convert time strings to datetime objects
    df.set_index("time", inplace=True)
    return df

def plot_heart_rate_for_day(date, label):
    """
    Plot heart rate data for a given day (intraday data) using Streamlit's line_chart.
    The JSON data is loaded from static/data for the given date and label.
    """
    dataset = load_heart_rate_data_for_day(date, label)
    if not dataset:
        st.warning("No heart rate data available.")
        return
    df = process_heart_rate_data(dataset)
    if df is None or df.empty:
        st.warning("No valid heart rate data to plot.")
        return
    df_plot = df.reset_index().rename(columns={"time": "Time", "value": "Heart Rate"})
    st.line_chart(
        df_plot,
        x="Time",
        y=["Heart Rate"],
        color=["#FF0000"]  # red color for the heart rate line
    )

def plot_heart_rate_for_range(start_date, end_date):
    """
    Plot heart rate summary data for a given date range (week trends) using Streamlit's line_chart.
    The JSON data is loaded from static/data for the given date range.
    (Assumes each day's summary is available under 'activities-heart' with a dateTime and a 'value'
     dict containing, for example, a 'restingHeartRate'.)
    """
    dataset = load_heart_rate_data_for_range(start_date, end_date)
    if not dataset:
        st.warning("No heart rate data available for the week.")
        return
    df = pd.DataFrame(dataset)
    # Extract a summary value (e.g. restingHeartRate) from the 'value' dictionary.
    if not df.empty and isinstance(df.loc[0, "value"], dict):
        df["Heart Rate"] = df["value"].apply(lambda x: x.get("restingHeartRate", None))
    else:
        df["Heart Rate"] = None
    df["Time"] = pd.to_datetime(df["dateTime"])
    df = df.dropna(subset=["Heart Rate"]).sort_values("Time")
    st.line_chart(
        df,
        x="Time",
        y=["Heart Rate"],
        color=["#FF0000"]
    )
