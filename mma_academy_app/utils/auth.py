"""
Authentication utilities for the Streamlit app.
Handles email-token based login and session management.
"""
import streamlit as st
import secrets  # For generating secure tokens
import smtplib  # For sending emails (requires SMTP server setup)
from email.mime.text import MIMEText

# ----------------------------------------------------------------------
# Attempt to import db utilities.  On older/newer code bases the
# "init_db_tables" function may have been renamed to just "init_db".
# ----------------------------------------------------------------------

# At the top of auth.py, replace the import block with:

try:
    from .db_utils import (
        init_db,  # Try new name first
        store_invite_token,
        verify_invite_token,
        DB_NAME,
        get_db_connection
    )
    # Alias for backward compatibility
    init_db_tables = init_db
except ImportError:
    try:
        from .db_utils import (
            init_db_tables,  # Try old name
            store_invite_token,
            verify_invite_token,
            DB_NAME,
            get_db_connection
        )
        # Alias for forward compatibility
        init_db = init_db_tables
    except ImportError:
        # If both fail, provide minimal imports
        from .db_utils import (
            store_invite_token,
            verify_invite_token,
            DB_NAME,
            get_db_connection
        )
        # Define a basic init function if neither exists
        def init_db():
            """Basic initialization function"""
            conn = get_db_connection()
            # Add any basic initialization here
            conn.close()
        init_db_tables = init_db

# Remove get_invite_status as it's not used/needed
def get_invite_status(email):
    """Compatibility function if needed"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT used FROM invites WHERE email=?", (email,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def init_db(): # Renamed from init_db_tables to match app.py call, though it calls the same
    """Ensure database and tables are initialized."""
    # st.write("DEBUG: init_db called from auth.py") # Debug line
    init_db_tables()

def send_access_code_email(email_to: str, access_code: str):
    """Sends the access code to the user via email."""
    try:
        smtp_config = st.secrets["smtp"]
        admin_email = st.secrets["admin"]["email"]
        
        msg = MIMEText(f"Hello,\n\nYour access code for the Indonesia MMA Academy Portal is: {access_code}\n\nThis code is valid for one-time use.\n\nIf you did not request this, please ignore this email.\n\nRegards,\nThe Indonesia MMA Academy Team")
        msg["Subject"] = "Your MMA Academy Access Code"
        msg["From"] = smtp_config["user"]
        msg["To"] = email_to

        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            server.starttls() # Secure the connection
            server.login(smtp_config["user"], smtp_config["password"])
            server.sendmail(smtp_config["user"], email_to, msg.as_string())
        st.success(f"Access code sent to {email_to}. Please check your inbox (and spam folder).")
        return True
    except KeyError as e:
        st.error(f"SMTP configuration error in secrets.toml: Missing key {e}. Email not sent.")
        return False
    except smtplib.SMTPAuthenticationError:
        st.error("SMTP Authentication Error: Username or password not accepted. Email not sent.")
        return False
    except smtplib.SMTPServerDisconnected:
        st.error("SMTP Server Disconnected: Could not connect to the SMTP server. Email not sent.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"SMTP Error: {e}. Email not sent.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while sending email: {e}")
        return False

def generate_and_send_access_code(email: str):
    """Generates a new access code, stores it, and attempts to email it."""
    # Check if email is the admin email or if it exists in invites (for re-sending)
    # For a new user, an admin would typically add their email to the invites table first.
    # However, the prompt implies admin can also login directly.
    is_admin_email = (email == st.secrets.get("admin", {}).get("email"))

    # For a robust system, you might want to check if the email is allowed to receive a code
    # (e.g., pre-invited by an admin). For now, let's assume admin can always get a code,
    # and other users must be in the invites table (even if token is expired/used).
    # The pasted_content_2.txt auth.py implies a check `if not c.fetchone() and email!=admin: return False`
    # which means non-admin emails must exist in invites. Let's replicate that logic.

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM invites WHERE email=?", (email,))
    existing_invite = cursor.fetchone()
    conn.close()

    if not existing_invite and not is_admin_email:
        st.error(f"Email {email} is not registered or invited. Please contact an administrator.")
        return False

    access_code = secrets.token_urlsafe(8)  # Generate a URL-safe token

    # Set role based on whether this is an admin email
    role = 'admin' if is_admin_email else 'user'

    try:
        if store_invite_token(email, access_code, role=role):
            if send_access_code_email(email, access_code):
                st.success(f"Access code sent to {email}. Please check your inbox (and spam folder).")
                return True
            else:
                st.warning(
                    "Access code generated and stored, but email sending failed. Please check SMTP settings or contact admin.")
                return False
        else:
            st.error("Failed to store access code. Please try again.")
            return False
    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")
        return False


def attempt_login(email: str, code: str):
    """Verifies the email and code against the database."""
    is_valid, role, user_id = verify_invite_token(email, code)
    if is_valid:
        st.session_state.authenticated = True
        st.session_state.user_email = email
        st.session_state.is_admin = (role == 'admin')
        st.session_state.user_id = user_id  # Store user_id for future use
        st.success("Login successful!")
        # st.balloons()
        return True
    else:
        st.error("Invalid email or access code, or code has expired.")
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.is_admin = False
        st.session_state.user_id = None
        return False
def check_authentication(show_sidebar_info=True):
    """Checks if user is authenticated. If not, stops the script and prompts login."""
    if not st.session_state.get("authenticated", False):
        st.warning("Access Denied! Please log in to view this page.")
        if show_sidebar_info:
            st.sidebar.info("Please log in using the Login page.")
        # st.page_link("pages/1_Login.py", label="Go to Login", icon="🔑") # Newer Streamlit
        st.stop() # Stop execution of the current page
    return True # User is authenticated

def logout():
    """Logs out the current user."""
    st.session_state.authenticated = False
    st.session_state.user_email = ""
    st.session_state.is_admin = False
    st.success("You have been logged out.")
    # Consider redirecting to login page or refreshing
    # st.experimental_rerun() # Use with caution, can cause loops if not handled well
    if "current_page" in st.session_state:
        del st.session_state.current_page # Reset current page if tracked
    st.switch_page("app.py") # Go to main app page, which will redirect to login

# Example of how an admin might add an invite (simplified, actual UI would be in Admin Panel)
# def add_new_invite(admin_email: str, new_user_email: str, make_admin: bool = False):
#     if not st.session_state.get("is_admin") or st.session_state.get("user_email") != admin_email:
#         st.error("Only admins can add new invites.")
#         return False
    
#     # Store a placeholder token; user will request a new one when they try to log in.
#     placeholder_token = "INVITED_BY_ADMIN_" + secrets.token_hex(4)
#     if store_invite_token(new_user_email, placeholder_token, is_admin=make_admin):
#         # Mark as used so user has to request a new one
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("UPDATE invites SET used = 1 WHERE email = ?", (new_user_email,))
#         conn.commit()
#         conn.close()
#         st.success(f"User {new_user_email} invited. They can now request an access code.")
#         return True
#     return False

if __name__ == '__main__':
    # For testing auth functions standalone (requires Streamlit context for secrets/session_state)
    print("Auth utilities module. Run within a Streamlit app context for full functionality.")
    # You would typically call init_db() from your main app.py or the login page.
    # init_db()

