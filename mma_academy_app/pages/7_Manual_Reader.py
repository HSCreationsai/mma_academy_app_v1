"""
Page 7: Manual Reader
Allows users to read the MMA training manual chapter by chapter (or page by page).
Includes Text-to-Speech (TTS) functionality.
"""
import streamlit as st
from utils.auth import check_authentication
import json
import os
from gtts import gTTS
import base64 # For embedding audio

# Define paths relative to the mma_academy_app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL_JSON_PATH = os.path.join(BASE_DIR, "data", "manual.json")
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "assets", "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

st.set_page_config(page_title="Training Manual - MMA Academy", layout="wide")

# Ensure user is authenticated
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to access the Training Manual.")
    st.switch_page("pages/1_Login.py")
    st.stop()

check_authentication()

st.title("📖 Indonesia MMA Training Manual")
st.markdown("Read the comprehensive training manual. Use the sidebar to navigate chapters/pages.")

# --- Load Manual Data --- 
@st.cache_data # Cache the loaded manual data
def load_manual_from_json(json_path=MANUAL_JSON_PATH):
    if not os.path.exists(json_path):
        st.error(f"Manual data file not found: {json_path}. Please ensure it has been generated.")
        st.caption("An administrator can try running the PDF to JSON conversion script from the Admin Panel or server-side.")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            manual_content = json.load(f)
        return manual_content
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from {json_path}: {e}. The file might be corrupted.")
        return None
    except Exception as e:
        st.error(f"Error loading manual data: {e}")
        return None

manual_data = load_manual_from_json()

if not manual_data:
    st.stop() # Stop if manual data couldn't be loaded

# --- Page/Chapter Selection --- 
# The manual.json is currently page-by-page. If it were chapter-based, logic would adapt.
page_keys = list(manual_data.keys())

if not page_keys:
    st.warning("The training manual appears to be empty. Please contact an administrator.")
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.header("Manual Navigation")
    selected_page_key = st.selectbox("Select Page/Chapter", page_keys, index=0, key="manual_page_select")

# --- Display Selected Page Content --- 
st.divider()
if selected_page_key and selected_page_key in manual_data:
    st.subheader(f"{selected_page_key}")
    page_text_content = manual_data[selected_page_key]
    
    # Display text content in a scrollable area
    with st.container(height=500, border=False):
        st.markdown(page_text_content)
    
    st.divider()
    
    # --- Text-to-Speech (TTS) --- 
    st.subheader("🔊 Read Aloud (Text-to-Speech)")
    if st.button(f"Generate Audio for {selected_page_key}", key=f"tts_button_{selected_page_key.replace(' ', '_')}"):
        if page_text_content.strip():
            with st.spinner(f"Generating audio for {selected_page_key}... This may take a moment."):
                try:
                    tts = gTTS(text=page_text_content, lang=	"en", slow=False) # Can try different languages if needed, e.g., "id" for Indonesian
                    # Sanitize filename
                    safe_filename_key = "".join(c if c.isalnum() else "_" for c in selected_page_key)
                    temp_audio_file = os.path.join(TEMP_AUDIO_DIR, f"audio_{safe_filename_key}.mp3")
                    tts.save(temp_audio_file)
                    
                    # Embed audio player
                    # st.audio(temp_audio_file, format="audio/mp3") # Standard way
                    # Custom embed to allow auto-cleanup and potentially better display
                    with open(temp_audio_file, "rb") as audio_file_obj:
                        audio_bytes = audio_file_obj.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success(f"Audio generated for {selected_page_key}. Press play above.")
                    
                    # Clean up the temporary audio file (optional, or manage temp files periodically)
                    # For simplicity in a multi-user or long-running app, cleanup is good.
                    # However, Streamlit's execution model might make direct cleanup tricky without session state tracking.
                    # A simple approach is to delete it after serving, but user might want to replay.
                    # For now, let's leave it and assume periodic cleanup or manage via admin.
                    # if os.path.exists(temp_audio_file):
                    #     os.remove(temp_audio_file)

                except ConnectionError as ce:
                    st.error(f"TTS Generation Failed: Could not connect to Google Translate services. Please check your internet connection. Error: {ce}") 
                except Exception as e:
                    st.error(f"Could not generate audio for {selected_page_key}. Error: {e}")
        else:
            st.warning("No text content on this page to read aloud.")
else:
    st.info("Select a page/chapter from the sidebar to view its content.")

st.divider()
st.caption("Use the sidebar to navigate through the manual. TTS feature requires an internet connection.")

