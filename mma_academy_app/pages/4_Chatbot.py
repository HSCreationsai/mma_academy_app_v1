# pages/4_Chatbot.py
from datetime import datetime

import streamlit as st
import time
import json
import torch
from transformers import pipeline
import csv
from pathlib import Path

try:
    from utils.auth import check_authentication
    from utils.rag_agent import (query_rag_agent,
                                 build_or_load_faiss_index, get_llm_pipeline, get_embedding_model)
    from utils.logging import log_query_entry
except ImportError as e:
    import streamlit as st

    st.error(f"Failed to import RAG or logging modules: {str(e)}")
    st.stop()

# Ensure user is authenticated
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to chat with Coach Olympus.")
    st.switch_page("pages/1_Login.py")
    st.stop()

check_authentication()

# --- PAGE CONFIG ---
st.set_page_config(page_title="🏛️ Coach Olympus - MMA AI Coach", layout="wide")
st.title("💬 Coach Olympus 🏛️: Your MMA AI Coach")
st.caption("Trained on the official Indonesia MMA Youth Excellence Academy training manual.")

# --- Initialize chat history if not present ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant",
         "content": "Hello! I am Coach Olympus. How can I help you with the MMA Training Manual today?"}
    ]

# --- GPU INFO - ADMIN ONLY ---
if st.session_state.get("is_admin", False):
    with st.expander("🧠 Developer System Info"):
        st.write(f"CUDA Available: {'✅ Yes' if torch.cuda.is_available() else '❌ No'}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            st.write(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.2f} GB")

        # Add model path info for admins
        st.write("**Model Configuration:**")
        st.write(f"- Primary model: `{st.secrets.get('llm', {}).get('model_path', 'Not configured')}`")
        st.write(f"- Medium fallback: `{st.secrets.get('model_fallbacks', {}).get('medium_model_path', 'None')}`")
        st.write(f"- Small fallback: `{st.secrets.get('model_fallbacks', {}).get('tiny_model_path', 'None')}`")
else:
    with st.expander("💡 About Coach Olympus"):
        st.markdown("""
        Coach Olympus 🏛️ uses advanced AI to read the official MMA training manual and answer your questions.

        - 📘 Answers are based **only** on the manual's content
        - ⚡ Best results with focused questions about training or rules
        - 🧠 Coach Olympus is still learning, so answers may evolve
        """)

# --- SIDEBAR OPTIONS ---
with st.sidebar:
    st.header("⚙️ Tools")
    if st.session_state.get("is_admin", False):
        if st.button("🔄 Rebuild RAG Index"):
            with st.spinner("Rebuilding FAISS index..."):
                _, _ = query_rag_agent("__rebuild_index__", top_k=0)
            st.success("✅ RAG index rebuilt!")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Chat history has been cleared. How can I help you today?"}
        ]
        st.rerun()

    # Optional: Translate output
    language_map = {
        "English": "en", "Bahasa Indonesia": "id", "中文 (Chinese)": "zh",
        "한국어 (Korean)": "ko", "Tagalog": "tl", "ไทย (Thai)": "th",
        "Português (Brazil)": "pt", "हिन्दी (Hindi)": "hi", "Русский": "ru",
        "Українська": "uk", "Polski": "pl"
    }
    lang_choice = st.selectbox("🌐 Translate answer to:", list(language_map.keys()), index=0)
    translate_to = language_map[lang_choice]
    translate = translate_to != "en"


# --- Initialize RAG components (can take a moment) ---
@st.cache_resource  # Cache the FAISS index, texts, and embedding model
def load_rag_components_once():
    with st.spinner("Coach Olympus is preparing the knowledge base... Please wait."):
        idx, texts = build_or_load_faiss_index()  # This will also handle saving/loading the FAISS index file
        emb_model = get_embedding_model()  # Ensure this is cached if it's heavy
    return idx, texts, emb_model


faiss_index, manual_texts, embedding_mdl = load_rag_components_once()

if faiss_index is None or manual_texts is None or embedding_mdl is None:
    st.error("Coach Olympus could not initialize properly. Please contact an administrator.")
    st.stop()

# --- LLM Model Path (from secrets) - Admin only warning ---
if not st.secrets.get("llm", {}).get("model_path") and st.session_state.get("is_admin", False):
    st.error("LLM model path is not configured properly. Please check secrets.toml.")

# Brief info about response times
st.info("""
💡 **AI Response Info**: Coach Olympus analyzes the MMA manual to answer your questions.
- Responses typically take 30-60 seconds to generate
- For faster results, ask specific questions about techniques
- If the AI is taking too long, you'll receive a partial answer with page references
- You can view the full content in the Manual Reader section
""")

# --- Display chat messages from history ---
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources from Manual", expanded=False):
                for i, source_page_key in enumerate(message["sources"]):
                    st.caption(f"Source {i + 1}: {source_page_key}")

# --- Get User Input ---
user_query = st.chat_input("💬 Ask Coach Olympus a question about the MMA training manual...")

if user_query:
    # Add user message to history
    st.session_state.chat_messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get AI response - Create a full container for the assistant message
    with st.chat_message("assistant"):
        # Create containers for the holographic coach animation and loading text
        hologram_container = st.empty()
        loading_container = st.empty()

        # Create more user-friendly loading messages that use coach terminology
        loading_messages = [
            "🥋 Youth Coach is analyzing your question...",
            "🏆 Pro Coach is searching the training manual...",
            "🧠 Master Coach Olympus is preparing your answer...",
            "⚡ Finding the most relevant techniques for you...",
            "🔍 Analyzing MMA concepts in the manual...",
            "🛠️ Crafting a comprehensive response..."
        ]

        # Add holographic coach visualization
        hologram_html = """
        <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
            <div class="hologram-coach" style="
                width: 100px;
                height: 100px;
                background: linear-gradient(45deg, #3a7bd5, #00d2ff);
                border-radius: 50%;
                box-shadow: 0 0 15px #00d2ff, 0 0 30px #3a7bd5;
                display: flex;
                justify-content: center;
                align-items: center;
                animation: pulse 2s infinite, rotate 8s linear infinite;
                position: relative;
                overflow: visible;
            ">
                <div style="
                    font-size: 40px;
                    color: white;
                    text-shadow: 0 0 10px white;
                ">🥋</div>
                <div style="
                    position: absolute;
                    width: 130px;
                    height: 130px;
                    border: 2px solid rgba(0, 210, 255, 0.3);
                    border-radius: 50%;
                    animation: expand 2s infinite;
                "></div>
            </div>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(0.95); }
                50% { transform: scale(1.05); }
                100% { transform: scale(0.95); }
            }
            @keyframes rotate {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes expand {
                0% { transform: scale(0.8); opacity: 1; }
                100% { transform: scale(1.2); opacity: 0; }
            }
        </style>
        """

        hologram_container.markdown(hologram_html, unsafe_allow_html=True)

        # Set up the loading animation with safer thread handling
        import threading
        import time
        import itertools
        import traceback

        # Store animation state in session
        if "loading_animation_running" not in st.session_state:
            st.session_state.loading_animation_running = False


        # Define safer loading animation function
        def loading_animation():
            try:
                st.session_state.loading_animation_running = True
                for message in itertools.cycle(loading_messages):
                    if not st.session_state.loading_animation_running:
                        break
                    try:
                        loading_container.info(message)
                        time.sleep(2.0)
                    except Exception:
                        # If UI update fails, just continue without crashing
                        time.sleep(2.0)
                        continue
            except Exception:
                # Safely exit if other errors occur
                pass


        # Start loading animation as daemon thread so it won't prevent app exit
        animation_thread = threading.Thread(target=loading_animation, daemon=True)
        animation_thread.start()

        # Initialize variables with default values in case of errors
        answer = "Sorry, I couldn't process your question."
        sources = []
        elapsed = 0
        response_content = answer

        try:
            # Redirect stdout to capture model loading messages if not in admin mode
            import io
            import sys

            original_stdout = None
            if not st.session_state.get("is_admin", False):
                original_stdout = sys.stdout
                sys.stdout = io.StringIO()  # Redirect stdout

            # Call RAG agent and measure time
            start = time.time()
            try:
                answer, sources = query_rag_agent(user_query)
                elapsed = round(time.time() - start, 2)
            finally:
                # Always restore stdout if redirected
                if original_stdout:
                    sys.stdout = original_stdout

            # Process the response
            if answer and answer.startswith("❌"):
                st.error(answer)
                response_content = answer
            else:
                # Optional translation
                if translate:
                    try:
                        with st.spinner("🔄 Translating..."):
                            translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{translate_to}")
                            translated = translator(answer, max_length=1024)[0]["translation_text"]
                            st.markdown(translated)
                            response_content = translated
                    except Exception as e:
                        # Fall back to original answer if translation fails
                        st.warning(f"Translation unavailable: {str(e)}")
                        st.markdown(answer)
                        response_content = answer
                else:
                    st.markdown(answer)
                    response_content = answer

                # Display source references with improved styling
                if sources:
                    with st.expander("📚 View Sources from Manual", expanded=False):
                        for i, source_page_key in enumerate(sources):
                            st.caption(f"📄 Source {i + 1}: {source_page_key}")

                # Show timing only for admins
                if st.session_state.get("is_admin", False):
                    st.caption(f"⏱️ Response generated in {elapsed} seconds")

        except Exception as e:
            # Handle any exceptions during processing
            error_message = f"❌ An error occurred: {str(e)}"
            st.error(error_message)
            answer = error_message
            response_content = answer

            # Log detailed error for admins
            if st.session_state.get("is_admin", False):
                st.error(f"Error details: {traceback.format_exc()}")

        finally:
            # Always stop the loading animation
            st.session_state.loading_animation_running = False

            # We don't join the thread as it's a daemon thread and will be terminated when the app exits
            # This avoids potential deadlocks if the thread is stuck

            # Clear the loading containers
            try:
                loading_container.empty()
                hologram_container.empty()
            except:
                pass  # Ignore errors if containers can't be emptied

        # Human feedback system with visual improvements
        st.write("")  # Add some spacing
        col1, col2 = st.columns(2)

        # Use session state for this specific answer's feedback
        feedback_key = f"feedback_{len(st.session_state.chat_messages)}"
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = {"submitted": False, "rating": None, "text": ""}

        # If feedback not yet submitted, show the buttons
        if not st.session_state[feedback_key]["submitted"]:
            with col1:
                if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_messages)}"):
                    st.session_state[feedback_key]["rating"] = "positive"
                    st.session_state[feedback_key]["submitted"] = True

                    # Log the positive feedback to CSV
                    _log_feedback(user_query, response_content, sources,
                                  "positive", "", lang_choice, elapsed)

                    st.success("Thanks for your feedback! This helps Coach Olympus improve.")
                    st.rerun()

            with col2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.chat_messages)}"):
                    st.session_state[feedback_key]["rating"] = "negative"

                    # Show text area for additional feedback
                    feedback_text = st.text_area(
                        "What could be improved?",
                        key=f"feedback_text_{len(st.session_state.chat_messages)}"
                    )

                    if st.button("Submit Feedback", key=f"submit_{len(st.session_state.chat_messages)}"):
                        st.session_state[feedback_key]["text"] = feedback_text
                        st.session_state[feedback_key]["submitted"] = True

                        # Log the negative feedback with text to CSV
                        _log_feedback(user_query, response_content, sources,
                                      "negative", feedback_text, lang_choice, elapsed)

                        st.success("Thanks for your detailed feedback! This helps Coach Olympus improve.")
                        st.rerun()
        else:
            # Show submitted feedback status
            if st.session_state[feedback_key]["rating"] == "positive":
                st.success("✅ You rated this answer as helpful. Thank you!")
            else:
                st.info("You provided feedback on this answer. Thank you for helping Coach Olympus improve!")

        # Add assistant response to history
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_content,
            "sources": sources
        })

        # Logging for Admins (technical details hidden from regular users)
        if st.session_state.get("is_admin"):
            log_query_entry(
                query=user_query,
                answer=response_content,
                sources=sources,
                model_name="olympus-coach-model",  # Generic name for logging
                duration=elapsed
            )


# Function to log feedback to CSV
def _log_feedback(query, answer, sources, rating, feedback_text, language, duration):
    """Log user feedback to CSV file for RLHF"""
    feedback_log_path = Path("logs/feedback_ratings.csv")
    feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = feedback_log_path.exists()

    with feedback_log_path.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "timestamp", "email", "question", "answer", "satisfaction",
                "feedback_text", "language", "duration", "sources"
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            st.session_state.get("user_email", "guest"),
            query,
            answer,
            rating,
            feedback_text,
            language,
            duration,
            "|".join(sources) if sources else ""
        ])


# --- Footer with system info (technical details only for admins) ---
st.divider()
st.caption(
    "**Coach Olympus** uses information solely from the official MMA training manual. For best results, ask specific questions about techniques or training methods.")

st.markdown("---")
st.markdown(
    f"<div style='text-align: right; font-size: 0.8em; color: gray;'>"
    f"{'🔒 Admin View' if st.session_state.get('is_admin') else '👤 User View'} | "
    "ℌ.𝔖𝔦𝔩𝔳𝔞 © 2025 MMA Academy"
    f"</div>", unsafe_allow_html=True
)