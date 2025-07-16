import logging
import uuid
import time
import requests
import streamlit as st
from functools import wraps


class LoggerManager:
    def __init__(self, log_file="user_activity.log"):
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("UserLogger")
        logger.setLevel(logging.INFO)

        # Remove old handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger

    def get_client_ip(self):
        if "client_ip" not in st.session_state:
            try:
                ip = requests.get("https://api.ipify.org").text
                st.session_state["client_ip"] = ip
            except:
                st.session_state["client_ip"] = "unknown"
        return st.session_state["client_ip"]

    def get_location_from_ip(self, ip):
        try:
            r = requests.get(f"https://ipinfo.io/{ip}/json")
            if r.status_code == 200:
                data = r.json()
                return f"{data.get('city', '')}, {data.get('region', '')}, {data.get('country', '')}"
        except:
            return "unknown"

    def log_user_activity(self, task_name="task", is_final=False):
        def wrapper(func):
            @wraps(func)
            def inner(*args, **kwargs):
                if "session_id" not in st.session_state:
                    st.session_state["session_id"] = str(uuid.uuid4())
                if "username" not in st.session_state:
                    st.session_state["username"] = "guest"

                username = st.session_state["username"]
                session_id = st.session_state["session_id"]
                ip = self.get_client_ip()
                location = self.get_location_from_ip(ip)

                start_time = time.time()

                self.logger.info(
                    f"[User: {username}] [Session: {session_id}] [IP: {ip}] [Location: {location}] - "
                    f"STARTED `{func.__name__}` [Task: {task_name}]"
                )

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.logger.error(
                        f"[User: {username}] [Session: {session_id}] - ERROR in `{func.__name__}`: {str(e)}"
                    )
                    raise
                finally:
                    duration = round(time.time() - start_time, 2)
                    self.logger.info(
                        f"[User: {username}] [Session: {session_id}] - COMPLETED `{func.__name__}` "
                        f"in {duration}s [Final: {is_final}]"
                    )

            return inner
        return wrapper
