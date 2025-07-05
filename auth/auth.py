import psycopg2
import bcrypt

# PostgreSQL Connection Settings
DB_PARAMS = {
    "dbname": "postgres",  # or your actual DB name if different
    "user": "dtbxnvdfsq@elderlydt-server",  # important: include server name after '@'
    "password": "v9$BeI$Xlx0YXYfv",
    "host": "elderlydt-server.postgres.database.azure.com",
    "port": "5432"
}

# Function to create a database connection
def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

# Function to hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Function to register a user in the database
def register_user(first_name, last_name, username, password, email, caregiver_type):
    hashed_pwd = hash_password(password)
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Step 1: Check if the table exists; if not, create it
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'users'
            )
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            cursor.execute("""
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    first_name VARCHAR(50) NOT NULL,
                    last_name VARCHAR(50) NOT NULL,
                    email VARCHAR(100) NOT NULL,
                    caregiver_type VARCHAR(20) NOT NULL,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

        # Step 2: Insert user
        cursor.execute("""
            INSERT INTO users (first_name, last_name, email, caregiver_type, username, password) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (first_name, last_name, email, caregiver_type, username, hashed_pwd))
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
    # grab first_name, last_name, and hashed password
    cursor.execute(
        "SELECT first_name, last_name, password FROM users WHERE username = %s",
        (username,),
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        return None

    first_name, last_name, hashed_pw = result
    if bcrypt.checkpw(password.encode(), hashed_pw.encode()):
        # return a dict with the bits you need
        return {
            "username":   username,
            "first_name": first_name,
            "last_name":  last_name
        }

    return None

def save_login_token(username, token):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET session_token = %s WHERE username = %s", (token, username))
    conn.commit()
    cursor.close()
    conn.close()

def check_token_valid(token):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE session_token = %s", (token,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None
