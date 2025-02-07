import streamlit as st
from auth.auth import login_user, register_user

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "show_register" not in st.session_state:
    st.session_state["show_register"] = False  # Default page is login

# Hide sidebar when not authenticated
if not st.session_state["authenticated"]:
    st.set_page_config(initial_sidebar_state="collapsed")

# Login Form
def login():
    st.title("Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.switch_page("pages/dashboard.py")  # ✅ Correct path: "pages/dashboard"
        else:
            st.error("Invalid username or password")

    if st.button("Register Here"):
        st.session_state["show_register"] = True

# Registration Form
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

# Main UI Logic (Routing)
if not st.session_state["authenticated"]:
    st.sidebar.empty()  # Hide sidebar when not authenticated
    if st.session_state["show_register"]:
        register()
    else:
        login()
else:
    st.switch_page("pages/dashboard.py")  # ✅ Correct path: "pages/dashboard"