"""
Page 5: Community Wall
Internal discussion forum for users.
"""
import streamlit as st
from utils.auth import check_authentication
from utils.db_utils import add_post, get_main_posts, get_replies_for_post
import datetime

st.set_page_config(page_title="Community Wall - MMA Academy", layout="wide")

# Ensure user is authenticated
if not st.session_state.get("authenticated", False):
    st.warning("Please log in to access the Community Wall.")
    st.switch_page("pages/1_Login.py")
    st.stop()

check_authentication()

st.title("💬 Community Wall")
st.markdown("Discuss training, share insights, and connect with fellow members of the Academy.")

st.divider()

# --- Option 1: Internal Discussion Forum (using SQLite) ---

# Form to create a new main post
st.subheader("✏️ Create a New Post")
with st.form("new_post_form", clear_on_submit=True):
    post_content = st.text_area("Your message:", height=100, placeholder="Share your thoughts or ask a question...")
    submit_post = st.form_submit_button("Post to Wall", use_container_width=True)

if submit_post:
    if post_content.strip():
        if add_post(user_email=st.session_state.user_email, content=post_content.strip()):
            st.success("Your post has been added to the wall!")
            st.rerun() # Refresh to show the new post
        else:
            st.error("Could not add your post. Please try again.")
    else:
        st.warning("Post content cannot be empty.")

st.divider()

st.subheader("🗣️ Recent Discussions")

main_posts = get_main_posts()

if not main_posts:
    st.info("No discussions yet. Be the first to start one!")
else:
    for post in main_posts:
        post_id = post["id"]
        user_email = post["user_email"]
        content = post["content"]
        timestamp = datetime.datetime.strptime(post["timestamp"], 	"%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y at %I:%M %p")

        with st.container(border=True):
            st.markdown(f"**Posted by:** {user_email} on {timestamp}")
            st.markdown(content)
            
            # Replies section
            replies = get_replies_for_post(post_id)
            if replies:
                with st.expander(f"View {len(replies)} Replies", expanded=False):
                    for reply in replies:
                        reply_user = reply["user_email"]
                        reply_content = reply["content"]
                        reply_time = datetime.datetime.strptime(reply["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y at %I:%M %p")
                        st.markdown(f"__{reply_user} ({reply_time}):__ {reply_content}")
                        st.markdown("---")
            
            # Form to add a reply
            with st.form(key=f"reply_form_{post_id}", clear_on_submit=True):
                reply_text = st.text_input("Write a reply...", key=f"reply_text_{post_id}", label_visibility="collapsed", placeholder="Write a reply...")
                submit_reply = st.form_submit_button("Reply")
            
            if submit_reply:
                if reply_text.strip():
                    if add_post(user_email=st.session_state.user_email, content=reply_text.strip(), parent_post_id=post_id):
                        st.success("Reply posted!")
                        st.rerun()
                    else:
                        st.error("Failed to post reply.")
                else:
                    st.warning("Reply cannot be empty.")
        st.markdown("&nbsp;") # Adds a little space between posts

# --- Option 2: Disqus Integration (Alternative, if preferred by user later) ---
# disqus_shortname = st.secrets.get("disqus", {}).get("shortname")
# if disqus_shortname:
#     st.subheader("Discussions (via Disqus)")
#     disqus_embed = f"""
#         <div id="disqus_thread"></div>
#         <script>
#             var disqus_config = function () {{
#                 this.page.url = "{st.experimental_get_query_params().get("page", ["community"])[0]}"; // Unique identifier for the page
#                 this.page.identifier = "mma_academy_community_wall"; // Unique identifier for the page
#             }};
#             (function() {{ // DON'T EDIT BELOW THIS LINE
#                 var d = document, s = d.createElement('script');
#                 s.src = 'https://{disqus_shortname}.disqus.com/embed.js';
#                 s.setAttribute('data-timestamp', +new Date());
#                 (d.head || d.body).appendChild(s);
#             }})();
#         </script>
#         <noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
#     """
#     st.components.v1.html(disqus_embed, height=800, scrolling=True)
# else:
#     st.info("Disqus integration is not configured in secrets.toml.")

st.divider()
st.caption("Maintain respectful and constructive discussions. All posts are subject to community guidelines.")

