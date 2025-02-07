import psycopg2
import bcrypt
import streamlit as st

# PostgreSQL Connection Settings
DB_PARAMS = {
    "dbname": "fitbit_db",
    "user": "fitbit_user",
    "password": "zia.mommand",
    "host": "localhost",
    "port": "5432"
}

# Function to create a database connection
def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

# Function to hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to register a user in the database
def register_user(first_name, last_name, username, password):
    hashed_pwd = hash_password(password)
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO users (first_name, last_name, username, password) 
            VALUES (%s, %s, %s, %s)
        """, (first_name, last_name, username, hashed_pwd))
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

# Function to check user login
def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if user:
        return bcrypt.checkpw(password.encode(), user[0].encode())
    return False