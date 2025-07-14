import logging
from functools import wraps
import streamlit as st
import uuid
import requests


class LoggerManager:
    def __init__(self, log_file=None):
        # Default to Azure-safe path
        if log_file is None:
            log_file = os.path.join("/home", "user_activity.log")
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Remove all handlers first
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])

        # Explicitly set up a new FileHandler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def get_client_ip(self):
        ip = st.session_state.get("client_ip")
        if not ip:
            try:
                ip = requests.get('https://api.ipify.org').text
                st.session_state["client_ip"] = ip
            except:
                ip = "unknown"
        return ip

    def get_location_from_ip(self, ip):
        try:
            response = requests.get(f"https://ipinfo.io/{ip}/json")
            if response.status_code == 200:
                data = response.json()
                return f"{data.get('city', '')}, {data.get('region', '')}, {data.get('country', '')}"
        except:
            pass
        return "unknown"

    def log_user_activity(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "session_id" not in st.session_state:
                st.session_state["session_id"] = str(uuid.uuid4())
            if "username" not in st.session_state:
                st.session_state["username"] = "guest"

            username = st.session_state["username"]
            session_id = st.session_state["session_id"]
            ip = self.get_client_ip()
            location = self.get_location_from_ip(ip)

            logging.info(
                f"[User: {username}] [Session: {session_id}] [IP: {ip}] [Location: {location}] "
                f"- Calling `{func.__name__}`"
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"[User: {username}] [Session: {session_id}] - Error in `{func.__name__}`: {e}")
                raise
            finally:
                logging.info(f"[User: {username}] [Session: {session_id}] - `{func.__name__}` completed.")
        return wrapper
