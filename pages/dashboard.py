import streamlit as st

# Redirect to login page if not authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.switch_page("main")  # ✅ Redirects back to login

st.title("Dashboard")
st.write(f"Welcome, {st.session_state['username']}! You have successfully logged in.")

# ✅ Sidebar Styled as a Menu (No New Tab)
with st.sidebar:
    st.markdown("### 📌 Navigation")
    st.markdown("---")

    if "menu_option" not in st.session_state:
        st.session_state["menu_option"] = "trends"  # Default page

    if st.button("💬 Chatbot", key="chatbot"):
        st.session_state["menu_option"] = "chatbot"

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
        st.switch_page("main.py")  # ✅ Redirect to login

# ✅ Render the Selected Page Content

if st.session_state["menu_option"] == "trends":
    st.subheader("📈 Trends")
    st.write("Trends and analytics visualization goes here...")
elif st.session_state["menu_option"] == "chatbot":
    st.subheader("💬 Chatbot")
    st.write("Chatbot interaction goes here...")

elif st.session_state["menu_option"] == "trends":
    st.subheader("📈 Trends")
    st.write("Trends and analytics visualization goes here...")

elif st.session_state["menu_option"] == "heart_model":
    st.subheader("❤️ Heart Model")
    st.write("3D Heart Model visualization goes here...")

elif st.session_state["menu_option"] == "respiratory_model":
    st.subheader("🫁 Respiratory Model")
    st.write("Respiratory Model visualization goes here...")
