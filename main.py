import streamlit as st
st.set_page_config(
    page_title="EDT System",
    page_icon="🩺",   
    layout="centered",
    initial_sidebar_state="collapsed"
)

import subprocess
import os
import time
import uuid
from streamlit_javascript import st_javascript
from auth.auth import login_user, register_user, save_login_token, check_token_valid
from pyngrok import ngrok

# ------------------------
# ✅ Start cors_server.py Automatically
# ------------------------
# Start an HTTP tunnel for port 8501 (or your Flask app port)
# public_url = ngrok.connect(8501, "http")
# print("Public URL:", public_url)
#CORS_SERVER_PATH = os.path.join(os.path.dirname(__file__), "static", "cors_server.py")

#def start_cors_server():
    #"""Start the CORS server in the background."""
    #print("Starting CORS Server...")
    #return subprocess.Popen(["python", CORS_SERVER_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start Flask CORS server
#cors_process = start_cors_server()
#time.sleep(2)  # Wait for the Flask server to initialize

# ------------------------
# ✅ Streamlit Session State Setup
# ------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "show_register" not in st.session_state:
    st.session_state.show_register = False # Default page is login

# Attempt to auto-restore session using token from localStorage
# if not st.session_state.authenticated:
#     token = st_javascript("window.localStorage.getItem('auth_token');")
#     if token:
#         username = check_token_valid(token)
#         if username:
#             st.session_state.authenticated = True
#             st.session_state.username = username
#             st.rerun()

# Hide sidebar when not authenticated
#if not st.session_state["authenticated"]:
    #st.set_page_config(initial_sidebar_state="collapsed")

# ------------------------
# ✅ Login Form
# ------------------------
def login():
    # Inject custom CSS
    st.markdown(
        """
        <style>
        .st-emotion-cache-1n76uvr {
            width: 100%;
            padding: 2rem;
        }
        .st-emotion-cache-1n76uvr h2 {
            text-align: center;
            color: #333333;
            margin-bottom: 1.5rem;
        }
        .st-emotion-cache-0 {
            text-align:center;
            width: 60%;
            margin-left: 18%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Centered containe
    st.markdown("## Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Buttons
    login_clicked = st.button(" 🔑 Login", type="primary", use_container_width=True)
    register_clicked = st.button("➕ Sign Up", type="primary", use_container_width=True)

    if login_clicked:
        # Have login_user return either None (bad) or a user-dict on success
        user = login_user(username, password)
        if user:
            token = str(uuid.uuid4())
            st.session_state["authenticated"] = True
            #save_login_token(user["username"], token)  # Save in DB
            #st_javascript(f"window.localStorage.setItem('auth_token', '{token}');")  # Save in browser

            st.session_state["username"]      = user["username"]
            st.session_state["first_name"]    = user["first_name"]
            st.session_state["last_name"]     = user["last_name"]
            st.success(f"Welcome, {user['first_name']}!")
            time.sleep(1)
            st.rerun()
            st.switch_page("pages/dashboard.py")
        else:
            st.error("Invalid username or password")

    if register_clicked:
        st.session_state.show_register = True

# ------------------------
# ✅ Registration Form
# ------------------------
def register():
    st.markdown(
        """
        <style>
        .st-emotion-cache-b95f0i {
            width: 100%;
            padding: 2rem;
        }
        .st-emotion-cache-b95f0i h2 {
            text-align: center;
            color: #333333;
            margin-bottom: 1.5rem;
        }
        .st-emotion-cache-b95f0i .stTextInput>div>div>input {
            padding: 0.75rem;
            border-radius: 4px;
        }
        .st-emotion-cache-0 {
            text-align:center;
            width: 60%;
            margin-left: 15%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Register")

    # Core caregiver registration fields
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email Address")
    caregiver_type = st.selectbox("Type of Caregiver", ["Family", "Volunteer", "Professional"])
    
    # Auth fields
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("➕ Register", type="primary", use_container_width=True):
        if password != confirm_password:
            st.warning("Passwords do not match.")
        elif not all([first_name, last_name, email, username, password]):
            st.warning("Please fill in all required fields.")
        else:
            success = register_user(first_name, last_name, username, password, email, caregiver_type)
            if success:
                st.success("User registered successfully! Now you can login.")
                st.session_state["show_register"] = False
            else:
                st.error("Username already exists or registration failed.")

    if st.button("🔑 Login", type="primary", use_container_width=True):
        st.session_state["show_register"] = False


# ------------------------
# ✅ Main UI Logic (Routing)
# ------------------------
if st.session_state.authenticated:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st_javascript("window.localStorage.removeItem('auth_token');")
        st.session_state.clear()
        st.experimental_rerun()
    st.switch_page("pages/dashboard.py")
else:
    if st.session_state.show_register:
        register()
    else:
        login()
