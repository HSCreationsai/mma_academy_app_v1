import streamlit as st
from utils.auth import init_db, check_authentication
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Indonesia MMA Academy", layout="wide", initial_sidebar_state="expanded")

# Initialize database and session state variables
if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# --- Page Styling (Optional) ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css") # If you have a style.css

# --- Main App Logic ---
if not st.session_state.authenticated:
    # If not authenticated, show a restricted view or redirect to login page (handled by Streamlit pages)
    st.switch_page("pages/1_Login.py")
else:
    with st.sidebar:
        st.image("assets/logo.png", width=150)
        st.markdown(f"### Welcome, {st.session_state.user_email}")
        st.markdown("---_" * 10)

        # Navigation Menu
        selected_page = option_menu(
            menu_title=None,  # "Main Menu"
            options=["Dashboard", "Video Center", "Manual Reader", "Chatbot", "Community Wall"],
            icons=["house-fill", "play-btn-fill", "book-half", "chat-dots-fill", "people-fill"], # https://icons.getbootstrap.com/
            menu_icon="cast",
            default_index=0,
            # orientation="horizontal", # Optional: for top menu
        )

        if st.session_state.is_admin:
            st.markdown("---_" * 10)
            admin_page = option_menu(
                menu_title="Admin Tools",
                options=["Admin Panel"],
                icons=["gear-fill"],
                menu_icon="tools",
                default_index=0,
            )
            if admin_page == "Admin Panel":
                st.switch_page("pages/6_Admin_Panel.py")

        st.markdown("---_" * 10)
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_email = ""
            st.session_state.is_admin = False
            st.success("Logged out successfully.")
            st.rerun()

    # Page routing based on selection
    if selected_page == "Dashboard":
        st.switch_page("pages/2_Dashboard.py")
    elif selected_page == "Video Center":
        st.switch_page("pages/3_Video_Center.py")
    elif selected_page == "Manual Reader":
        st.switch_page("pages/7_Manual_Reader.py")
    elif selected_page == "Chatbot":
        st.switch_page("pages/4_Chatbot.py")
    elif selected_page == "Community Wall":
        st.switch_page("pages/5_Community.py")

    # Fallback to Dashboard if no page is actively selected (e.g. on first load after login)
    # This is usually handled by Streamlit's multipage app structure if app.py is the main landing
    # and other .py files are in 'pages/' directory.
    # The first page alphabetically in 'pages/' is often the default if not specified.
    # To ensure Dashboard is the default after login, we can explicitly call it if current page is app.py
    # However, with st.switch_page, this logic is more explicit.
    # For the initial load after login, if selected_page is already Dashboard, it will switch to it.
    # If app.py is loaded directly and authenticated, it should show the dashboard content or switch to it.
    # The current structure with st.switch_page should handle this.
    # To be absolutely sure, we can add a check here, but it might be redundant.
    # For now, let's assume the option_menu default and switch_page handle it.

    # Display a placeholder if on app.py directly after login (should be redirected by switch_page)
    # st.title("Indonesia MMA Youth Excellence Academy")
    # st.write("Select a module from the sidebar to get started.")

