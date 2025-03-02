import streamlit as st
import subprocess
import os
import time
from auth.auth import login_user, register_user
from pyngrok import ngrok

# ------------------------
# ✅ Start cors_server.py Automatically
# ------------------------
# Start an HTTP tunnel for port 8501 (or your Flask app port)
# public_url = ngrok.connect(8501, "http")
# print("Public URL:", public_url)
CORS_SERVER_PATH = os.path.join(os.path.dirname(__file__), "static", "cors_server.py")

def start_cors_server():
    """Start the CORS server in the background."""
    print("Starting CORS Server...")
    return subprocess.Popen(["python", CORS_SERVER_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start Flask CORS server
cors_process = start_cors_server()
time.sleep(2)  # Wait for the Flask server to initialize

# ------------------------
# ✅ Streamlit Session State Setup
# ------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "show_register" not in st.session_state:
    st.session_state["show_register"] = False  # Default page is login

# Hide sidebar when not authenticated
if not st.session_state["authenticated"]:
    st.set_page_config(initial_sidebar_state="collapsed")

# ------------------------
# ✅ Login Form
# ------------------------
def login():
    st.title("Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.switch_page("pages/dashboard.py")  # ✅ Redirect to dashboard
        else:
            st.error("Invalid username or password")

    if st.button("Register Here"):
        st.session_state["show_register"] = True

# ------------------------
# ✅ Registration Form
# ------------------------
def register():
    st.title("Register")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Register"):
        if register_user(first_name, last_name, username, password):
            st.success("User registered successfully! Now you can login.")
            st.session_state["show_register"] = False
        else:
            st.error("Username already exists!")

    if st.button("Back to Login"):
        st.session_state["show_register"] = False

# ------------------------
# ✅ Main UI Logic (Routing)
# ------------------------
if not st.session_state["authenticated"]:
    st.sidebar.empty()  # Hide sidebar when not authenticated
    if st.session_state["show_register"]:
        register()
    else:
        login()
else:
    st.switch_page("pages/dashboard.py")  # ✅ Redirect to dashboard
