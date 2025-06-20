"""
Page 1: Login Page
Handles user login using email and a one-time access code.
"""
import streamlit as st
from utils.auth import generate_and_send_access_code, attempt_login, init_db

st.set_page_config(page_title="Login - MMA Academy", layout="centered")

# Initialize DB if not already done (e.g., if user lands here directly)
if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    st.success(f"Already logged in as {st.session_state.user_email}.")
    st.page_link("app.py", label="Go to Dashboard", icon="🏠")
    st.stop()

st.title("🥋 Indonesia MMA Academy Portal Login")
st.markdown("Access your training resources and community.")

login_tab, request_code_tab = st.tabs(["Login with Code", "Request Access Code"])

with login_tab:
    st.subheader("Enter Your Credentials")
    with st.form("login_form"):
        email_login = st.text_input("Email Address", key="email_login", placeholder="your.email@example.com")
        access_code_login = st.text_input("Access Code", key="access_code_login", type="password", placeholder="Enter code from email")
        submit_login = st.form_submit_button("Login", use_container_width=True)

    if submit_login:
        if not email_login or not access_code_login:
            st.error("Please enter both email and access code.")
        else:
            if attempt_login(email_login, access_code_login):
                # Login successful, redirect to the main app page (app.py will handle further routing)
                st.switch_page("app.py")
            # Error messages are handled within attempt_login

with request_code_tab:
    st.subheader("Request a New Access Code")
    st.markdown("If you are a registered user or have been invited, enter your email below to receive a one-time access code.")
    with st.form("request_code_form"):
        email_request = st.text_input("Email Address", key="email_request", placeholder="your.email@example.com")
        submit_request_code = st.form_submit_button("Send Access Code", use_container_width=True)

    if submit_request_code:
        if not email_request:
            st.error("Please enter your email address.")
        else:
            # generate_and_send_access_code will show success/error messages
            generate_and_send_access_code(email_request)

st.markdown("---_" * 10)
st.caption("Ensure you have access to the email address provided. Access codes are single-use and expire.")

