"""
Page 2: Dashboard
Displays a personalized welcome and progress summary (placeholder for now).
"""
import streamlit as st
from utils.auth import check_authentication

st.set_page_config(page_title="Dashboard - MMA Academy", layout="wide")

# Ensure user is authenticated before accessing the page
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to view the Dashboard.")
    st.switch_page("pages/1_Login.py") # Redirect to login
    st.stop()

check_authentication() # Double check, and stops if not authenticated

# --- Dashboard Content ---
st.title(f"🥋 Welcome to Your Dashboard, {st.session_state.user_email}!")
st.markdown("Here you can find an overview of your activities, progress, and important updates from the Academy.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Quick Actions")
    if st.button("Go to Video Center", use_container_width=True, type="primary"):
        st.switch_page("pages/3_Video_Center.py")
    if st.button("Read the Training Manual", use_container_width=True):
        st.switch_page("pages/7_Manual_Reader.py")
    if st.button("Ask the AI Chatbot", use_container_width=True):
        st.switch_page("pages/4_Chatbot.py")
    if st.button("Visit Community Wall", use_container_width=True):
        st.switch_page("pages/5_Community.py")
    if st.session_state.get("is_admin", False):
        if st.button("Access Admin Panel", use_container_width=True):
            st.switch_page("pages/6_Admin_Panel.py")

with col2:
    st.subheader("📈 Your Progress (Placeholder)")
    st.info("This section will display your training progress, completed modules, and other relevant statistics. Feature coming soon!")
    # Example placeholders for progress
    st.slider("Overall Manual Completion", 0, 100, 25, disabled=True)
    st.metric("Videos Watched", "12 / 50", delta="2 new", delta_color="off")
    st.metric("Last Login", "May 07 2025, 10:00 AM", delta_color="off")

st.divider()

st.subheader("📢 Academy Announcements (Placeholder)")
st.warning("No new announcements at this time. Check back later!")

# --- Footer or additional info ---
st.caption("Remember to stay disciplined and consistent with your training!")

