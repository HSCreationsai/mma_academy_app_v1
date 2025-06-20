
"""
Page 6: Admin Panel
Allows admins to manage content like videos, PDF resources, and user invites.
"""
import streamlit as st
import os
import secrets  # Add this import for token generation

# Try to import authentication utilities
try:
    from utils.auth import check_authentication, generate_and_send_access_code
except ImportError:
    def check_authentication(*args, **kwargs):
        st.error("Authentication functionality is not available")
        st.stop()
    def generate_and_send_access_code(*args, **kwargs):
        st.error("Access code generation is not available")
        return False

# Try to import secure filename utility
try:
    from werkzeug.utils import secure_filename
except ImportError:
    def secure_filename(filename):
        return filename.replace(" ", "_")

# Try to import invite management functions
try:
    from utils.video_utils import is_video_url_reachable

    from utils.db_utils import store_invite_token, get_all_invites
except ImportError:
    def store_invite_token(*args, **kwargs):
        st.error("Invite management functionality is not available")
        return False
    def get_all_invites(): return []

# Try to import video management functions
try:
    from utils.db_utils import add_video, get_all_videos, delete_video
except ImportError:
    def add_video(*args, **kwargs):
        st.error("Video management functionality is not available")
        return None
    def get_all_videos(): return []
    def delete_video(*args, **kwargs): return False

# Try to import resource management functions
try:
    from utils.pdf_to_json import convert_pdf_to_json
    from utils.rag_agent import build_or_load_faiss_index
    from utils.db_utils import add_resource, get_all_resources, delete_resource
except ImportError:
    def add_resource(*args, **kwargs):
        st.error("Resource management functionality is not available")
        return None
    def get_all_resources(): return []
    def delete_resource(*args, **kwargs): return False

# Define paths relative to the mma_academy_app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADED_RESOURCES_DIR = os.path.join(BASE_DIR, "assets", "uploads")
os.makedirs(UPLOADED_RESOURCES_DIR, exist_ok=True)

st.set_page_config(page_title="Admin Panel - MMA Academy", layout="wide")

# Ensure user is authenticated AND is an admin
if not st.session_state.get("authenticated", False) or not st.session_state.get("is_admin", False):
    st.error("Access Denied. You must be an administrator to view this page.")
    if not st.session_state.get("authenticated", False):
        st.switch_page("pages/1_Login.py") # Redirect to login if not logged in at all
    else:
        st.page_link("app.py", label="Go to Dashboard", icon="🏠")
    st.stop()

check_authentication() # Redundant if above check passes, but good for safety
if not st.session_state.is_admin:
    st.error("You do not have administrative privileges.")
    st.stop()

st.title("🛠️ Admin Panel")
st.markdown(f"Logged in as Admin: **{st.session_state.user_email}**")

st.divider()

admin_tabs = st.tabs(["Manage Videos", "Manage PDF Resources", "Manage User Invites", "System Info"])

with admin_tabs[0]:
    st.subheader("🎬 Manage Training Videos")
    with st.form("add_video_form", clear_on_submit=True):
        st.write("Add a new video by providing its title and URL (e.g., YouTube, Vimeo link).")
        video_title = st.text_input("Video Title")
        video_url = st.text_input("Video URL (must be a direct link or embeddable URL)")
        submit_add_video = st.form_submit_button("Add Video")

    if st.button("🔄 Revalidate All Videos"):
        st.info("Checking all stored video links...")
        failures = []
        for v in get_all_videos():
            if not is_video_url_reachable(v["url"]):
                failures.append(f"{v['title']} → {v['url']}")
        if failures:
            st.error("⚠️ Some video links appear to be unreachable:")
            for f in failures:
                st.markdown(f"- {f}")
        else:
            st.success("✅ All videos are reachable.")

    if submit_add_video:
        if video_title and video_url:
            if add_video(video_title, video_url, st.session_state.get("user_email", "unknown_user")):
                st.success(f"Video '{video_title}' added successfully!")
                st.rerun()
            # Error is handled in add_video
        else:
            st.warning("Please provide both title and URL for the video.")

    st.markdown("---_" * 10)
    st.write("**Available Videos:**")
    videos = get_all_videos()

    if not videos:
        st.info("No videos uploaded yet.")
    else:
        for video in videos:
            title = video.get("title", "Untitled")
            url = video.get("url", "No URL Provided")
            video_id = video.get("id", f"unknown_{title}")

            col1, col2, col3 = st.columns([3, 4, 1])
            with col1:
                st.write(f"**{title}**")
            with col2:
                st.caption(f"URL: {url}")
            with col3:
                if st.button("Delete", key=f"del_video_{video_id}", type="secondary"):
                    if delete_video(video_id):
                        st.success(f"Video '{title}' deleted.")
                        st.rerun()
            st.markdown("---_" * 5)

with admin_tabs[1]:
    st.subheader("📄 Manage PDF Resources")
    st.markdown("Upload additional PDF documents (e.g., articles, supplementary guides). These are separate from the main training manual.")
    
    with st.form("upload_pdf_form", clear_on_submit=True):
        pdf_title = st.text_input("PDF Resource Title")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        submit_upload_pdf = st.form_submit_button("Upload PDF Resource")

    if submit_upload_pdf:
        if pdf_title and uploaded_file is not None:
            file_name = secure_filename(uploaded_file.name)
            file_path = os.path.join(UPLOADED_RESOURCES_DIR, file_name)
            
            # Save the uploaded file
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to database
                if add_resource(pdf_title, file_name, file_path, st.session_state.user_email):
                    st.success(f"PDF resource 	'{pdf_title}	' ({file_name}) uploaded and saved successfully!")
                    st.rerun()
                else:
                    # If DB add fails, try to remove the orphaned file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    st.error("Failed to add PDF resource to database. File not saved.")
            except Exception as e:
                st.error(f"Error saving uploaded PDF: {e}")
        else:
            st.warning("Please provide a title and select a PDF file to upload.")

    st.markdown("---_" * 10)
    st.write("**Uploaded PDF Resources:**")
    resources = get_all_resources()
    if not resources:
        st.info("No PDF resources uploaded yet.")
    else:
        for res in resources:
            col1, col2, col3 = st.columns([3,4,1])
            with col1:
                st.write(f"**{res["title"]}**") 
            with col2:
                # Provide a link to view/download the PDF. Streamlit doesn't directly embed PDFs from local paths easily without workarounds.
                # For simplicity, we can offer a download link or inform admin where it's stored.
                # A proper solution might involve serving these files via a simple HTTP server if direct embedding is needed.
                st.caption(f"Filename: {res["file_name"]}") # Path: {res["file_path"]}
                # st.page_link(res["file_path"], label=f"Open {res["file_name"]}") # This won't work directly for local files
                # A download button is more robust for local files:
                try:
                    with open(res["file_path"], "rb") as fp:
                        st.download_button(
                            label=f"Download 	{res["file_name"]}	",
                            data=fp,
                            file_name=res["file_name"],
                            mime="application/pdf",
                            key=f"download_res_{res["id"]}"
                        )
                except FileNotFoundError:
                    st.error(f"File {res["file_name"]} not found at path. It might have been moved or deleted.")
                except Exception as e:
                    st.error(f"Could not prepare download for {res["file_name"]}: {e}")
            with col3:
                if st.button("Delete", key=f"del_res_{res["id"]}", type="secondary"):
                    file_to_delete_path = res["file_path"]
                    if delete_resource(res["id"]):
                        st.success(f"Resource 	'{res["title"]}	' deleted from database.")
                        # Also delete the actual file from server
                        if os.path.exists(file_to_delete_path):
                            try:
                                os.remove(file_to_delete_path)
                                st.info(f"File {res["file_name"]} deleted from server.")
                            except Exception as e:
                                st.warning(f"Could not delete file {res["file_name"]} from server: {e}")
                        st.rerun()
                    # Error handled in delete_resource
            st.markdown("---_" * 5)

with admin_tabs[2]:
    st.subheader("🔑 Manage User Invites")
    st.markdown("Invite new users by providing their email address. They will receive an access code to log in. Admins can also be designated here.")

    with st.form("invite_user_form", clear_on_submit=True):
        new_user_email = st.text_input("New User Email Address")
        is_new_admin = st.checkbox("Make this user an Administrator?")
        submit_invite = st.form_submit_button("Send Invite & Generate Access Code")

    if submit_invite:
        if new_user_email:
            # The generate_and_send_access_code function from auth.py handles storing the token and sending the email.
            # It needs the email to be in the invites table first if it's not an admin, or it's the admin email itself.
            # For a new invite, we first add/update the invite record, then trigger the email.
            
            # Step 1: Store/Update the invite intention (token will be placeholder initially or overwritten)
            # The `generate_and_send_access_code` in auth.py already handles storing the token.
            # We just need to ensure it can be called for a new user.
            # The current `generate_and_send_access_code` checks if a non-admin email exists in invites.
            # So, we must first add the email to invites with a temporary token if it's a brand new user.
            
            # Simplified: Let's use a direct function to add/update invite status then send code.
            temp_token = "TEMP_INVITE_" + secrets.token_hex(4)
            if store_invite_token(new_user_email, temp_token, is_admin=is_new_admin):
                 st.info(f"Invite for {new_user_email} registered. Now attempting to send access code email...")
                 # Now call the function that generates a new token and sends it
                 if generate_and_send_access_code(new_user_email): # This will overwrite temp_token and send email
                     st.success(f"Access code email process initiated for {new_user_email}.")
                     st.rerun()
                 else:
                     st.error(f"Failed to send access code email to {new_user_email}, but user is registered. They can try 'Request Access Code' on login page.")
            else:
                st.error(f"Failed to register invite for {new_user_email}.")
        else:
            st.warning("Please enter the email address for the new user.")

    st.markdown("---_" * 10)
    st.write("**Existing Invites/Users:**")
    invites = get_all_invites()
    if not invites:
        st.info("No users invited yet.")
    else:
        for inv in invites:
            st.markdown(f"- **Email:** {inv.get["email"]}")
            st.markdown(f"  - **Admin:** {'Yes' if inv.get('is_admin') else 'No'}")
            st.markdown(f"  - **Code Used:** {'Yes' if inv.get('used') else 'No'}")
            st.markdown(f"  - **Invite Date:** {inv.get["created_at"]}")
            # Add option to resend code or revoke invite if needed (more complex)
            st.markdown("---_" * 5)

with admin_tabs[3]:
    st.subheader("⚙️ System Information")
    st.markdown("Basic system and application status.")
    # Placeholder for system info - could include DB stats, number of users, etc.
    st.info("This section is a placeholder for future system diagnostics and information.")
    if st.button("Re-initialize RAG Index (if manual.json updated)"):
        with st.spinner("Re-building FAISS index from manual.json..."):

            # Clear cached index first
            st.cache_resource.clear()
            # global _index, _texts # from rag_agent, cannot directly modify here
            # _index = None
            # _texts = None
            # This is tricky because the cache is on the function in rag_agent.
            # A better way would be a dedicated function in rag_agent to force rebuild.
            # For now, clearing cache_resource might work if load_rag_components is called again.
            st.success("Attempted to clear RAG cache. Index will rebuild on next Chatbot load or query.")
            st.warning("For a full rebuild, you might need to restart the Streamlit app after clearing cache if issues persist.")

    if st.button("Run PDF to JSON Conversion (manual_draft.pdf -> manual.json)"):

        st.info("Attempting to convert `assets/manual_draft.pdf` to `data/manual.json`...")
        with st.spinner("Converting PDF..."):
            if convert_pdf_to_json():
                st.success("PDF to JSON conversion successful. RAG index will need to be rebuilt (see above or restart app).")
            else:
                st.error("PDF to JSON conversion failed. Check console logs if running locally.")

st.divider()
st.caption("Manage application content and users from this panel.")

