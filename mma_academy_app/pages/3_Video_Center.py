"""
Page 3: Video Center
Allows users to view admin-curated training videos.
"""
import streamlit as st

# --- AUTH ---
try:
    from utils.auth import check_authentication
except ImportError as e:
    st.error(f"Failed to import authentication module: {str(e)}")
    st.warning("Authentication unavailable. Contact admin.")
    st.stop()

# --- DB ---
try:
    from utils.db_utils import get_all_videos
except ImportError as e:
    st.error(f"Failed to import database utilities: {str(e)}")
    st.warning("Video retrieval unavailable. Contact admin.")
    st.stop()

st.set_page_config(page_title="Video Center - MMA Academy", layout="wide")

# --- AUTH CHECK ---
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to view the Video Center.")
    st.switch_page("pages/1_Login.py")
    st.stop()

check_authentication()

# --- UI ---
st.title("🎬 MMA Training Video Center")
st.markdown("Browse and watch curated training videos to enhance your skills.")
st.divider()

videos = get_all_videos()

if not videos:
    st.info("No training videos uploaded yet. Please check back later.")
else:
    st.subheader(f"Available Videos ({len(videos)})")

    for video in videos:
        title = video.get("title", "Untitled")
        url = video.get("url", "")
        uploaded_by = video.get("uploaded_by", "Unknown")
        uploaded_at = video.get("uploaded_at", "N/A")[:10]
        video_id = video.get("id", "unknown")

        expander_label = f"**{title}** (Uploaded by: {uploaded_by} on {uploaded_at})"
        with st.expander(expander_label, expanded=False):
            st.markdown(f"#### {title}")

            if "youtube.com/watch?v=" in url or "youtu.be/" in url:
                try:
                    st.video(url)
                except Exception as e:
                    st.error(f"Could not embed video. Error: {e}")
                    st.caption(f"URL: {url}")
            elif url.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                st.warning("Direct video streaming may be limited by browser support.")
                try:
                    st.video(url)
                except Exception as e:
                    st.error(f"Could not load video file. Error: {e}")
                    st.caption(f"Video URL: {url}")
            else:
                st.warning(f"Unsupported video URL format.")
                st.markdown(f"[🔗 Open Video]({url})")

            st.caption(f"Video ID: {video_id} | Uploaded: {uploaded_at}")

st.divider()
st.caption("For optimal viewing, ensure a stable internet connection.")

st.markdown("---")
st.markdown(
    f"<div style='text-align: right; font-size: 0.8em; color: gray;'>"
    f"{'🔒 Admin View' if st.session_state.get('is_admin') else '👤 User View'} | "
    "ℌ.𝔖𝔦𝔩𝔳𝔞 © 2025 MMA Academy"
    f"</div>", unsafe_allow_html=True
)
