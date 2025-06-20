"""
Streamlit Page for Coach Certification Module (Levels 1, 2, 3).
Allows users to take quizzes and earn certificates, with DB integration.
"""
import streamlit as st
from streamlit_option_menu import option_menu
import datetime
import os
import random  # Keep for sampling if a quiz def uses it
import sys

# Adjust import paths for utils if necessary
PROJECT_ROOT_DIR_PAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT_DIR_PAGE)

# Try importing auth utilities
try:
    from utils.auth import get_current_user_email, is_user_logged_in, get_current_user_id
except ImportError as e:
    st.error(f"Failed to import authentication utilities: {e}")
    st.stop()

# Try importing quiz engine
try:
    from utils.quiz_engine import QuizEngine
except ImportError as e:
    st.error(f"Failed to import quiz engine: {e}")
    st.stop()

# Try importing certificate generator
try:
    from utils.certificate_generator import generate_certificate_pdf
except ImportError as e:
    st.error(f"Failed to import certificate generator: {e}")
    st.stop()

# Try importing database utilities
try:
    from utils.db_utils import (
        get_db_connection,
        get_user_by_email,
        get_active_quizzes_by_level,
        get_questions_for_quiz as get_questions_for_quiz_from_db,
        save_user_quiz_attempt,
        get_user_level_progress,
        update_user_level_progress,
        save_user_certificate,
        get_user_certificate,
        get_user_quiz_attempts
    )
except ImportError as e:
    st.error(f"Failed to import database utilities: {e}")
    st.stop()

PAGE_TITLE = "Coach Olympus Certification Program"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- Helper to get user_id from email ---
@st.cache_data(ttl=3600) # Cache for an hour
def fetch_user_id(email: str):
    user_db_data = get_user_by_email(email)
    if user_db_data:
        return user_db_data["id"]
    return None

def display_quiz(engine: QuizEngine, user_id: int):
    st.subheader(f"Quiz: {engine.quiz_definition["title"]}")
    st.caption(engine.quiz_definition.get("description", ""))
    if engine.quiz_definition.get("time_limit_minutes"):
        # Basic timer display - more advanced timer would need client-side JS or frequent reruns
        # For now, just inform the user.
        st.info(f"Time Limit: {engine.quiz_definition["time_limit_minutes"]} minutes. Please manage your time.")
    st.markdown("---")

    current_q = engine.get_current_question()
    if not current_q:
        st.error("Error: Could not load current question.")
        return

    st.markdown(f"**Question {engine.current_question_index + 1} of {len(engine.current_quiz_questions)}**")
    st.markdown(f"*{current_q["text"]}* ({current_q.get("marks",1)} marks)")

    # Display image if available
    if current_q.get("image_url"):
        # Assuming image_url is a path relative to assets or an absolute web URL
        # For local files, ensure they are accessible (e.g., in a static folder if deployed)
        # For this example, let's assume it's a web URL or correctly pathed local file for Streamlit
        st.image(current_q["image_url"], width=300) # Adjust width as needed

    # Display video if available (Streamlit supports YouTube, Vimeo, or local files)
    if current_q.get("video_url"):
        st.video(current_q["video_url"])

    if current_q["question_type"] == "mcq":
        # Options should already be parsed into a list of dicts by db_utils.get_questions_for_quiz
        options_list = current_q.get("options", []) 
        if not options_list:
            st.error("MCQ question has no options defined!")
            return
        options_dict = {opt["id"]: opt["text"] for opt in options_list}
        user_choice_id = st.radio("Select your answer:", options_dict.keys(), format_func=lambda x: options_dict[x], key=f"q_{current_q["id"]}")
        engine.record_answer(current_q["id"], user_choice_id)
    elif current_q["question_type"] == "true_false":
        tf_options = {"True": "True", "False": "False"} # Standard True/False
        user_choice_tf = st.radio("Select your answer:", list(tf_options.keys()), key=f"q_{current_q["id"]}")
        engine.record_answer(current_q["id"], user_choice_tf)
    # Add other question types (video_mcq) here if they have distinct rendering logic

    col1, col2, col3 = st.columns([1,5,1])
    with col1:
        if engine.current_question_index > 0:
            if st.button("⬅️ Previous", use_container_width=True, key=f"prev_q_{engine.current_quiz_id}"):
                engine.previous_question()
                st.rerun()
    with col3:
        if engine.current_question_index < len(engine.current_quiz_questions) - 1:
            if st.button("Next ➡️", use_container_width=True, key=f"next_q_{engine.current_quiz_id}"):
                engine.next_question()
                st.rerun()
        else:
            if st.button("🏁 Submit Quiz", type="primary", use_container_width=True, key=f"submit_q_{engine.current_quiz_id}"):
                st.session_state.quiz_submitted = True
                st.rerun()

def display_quiz_results(engine: QuizEngine, user_id: int):
    st.subheader("Quiz Results")
    score_details = engine.calculate_score()
    is_passed = engine.is_quiz_passed(score_details["percentage"])
    quiz_level = engine.quiz_definition["level"]

    st.metric("Your Score", f"{score_details["percentage"]}%", delta=f"{score_details["score_achieved"]}/{score_details["total_marks_possible"]} marks")
    if is_passed:
        st.success(f"Congratulations! You passed the Level {quiz_level} quiz.")
        # Update user level progress
        update_user_level_progress(user_id, quiz_level, status="completed_online_tests", online_tests_completed_at=datetime.datetime.now())
    else:
        st.error(f"Unfortunately, you did not pass. Required: {engine.quiz_definition["passing_score_percentage"]}%")
        # Potentially update progress to reflect failed attempt if needed, or just log attempt
    
    # Save attempt to DB
    user_attempts_for_this_quiz = get_user_quiz_attempts(user_id, engine.current_quiz_id)
    current_attempt_number = len(user_attempts_for_this_quiz) + 1
    save_user_quiz_attempt(user_id, engine.current_quiz_id, current_attempt_number, score_details, is_passed, engine.user_answers)

    if st.button("Back to Certification Overview", key=f"back_overview_{engine.current_quiz_id}"): 
        del st.session_state.active_quiz_engine
        del st.session_state.current_quiz_id
        if "quiz_submitted" in st.session_state: del st.session_state.quiz_submitted
        st.rerun()

    with st.expander("Review Your Answers"):
        for i, q_data in enumerate(engine.current_quiz_questions):
            st.markdown(f"**Q{i+1}: {q_data["text"]}**")
            user_ans_id = engine.user_answers.get(q_data["id"])
            correct_ans_id = q_data["correct_answer_id"]
            
            options_list = q_data.get("options", [])
            options_dict = {opt["id"]: opt["text"] for opt in options_list} if q_data["question_type"] == "mcq" else {"True":"True", "False":"False"}
            
            user_ans_text = options_dict.get(user_ans_id, str(user_ans_id))
            correct_ans_text = options_dict.get(correct_ans_id, str(correct_ans_id))

            if user_ans_id == correct_ans_id:
                st.markdown(f"<span style=\'color:green\	places'>Your answer: {user_ans_text} (Correct)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style=\'color:red\	places'>Your answer: {user_ans_text} (Incorrect)</span>", unsafe_allow_html=True)
                st.markdown(f"<span style=\'color:blue\	places'>Correct answer: {correct_ans_text}</span>", unsafe_allow_html=True)
            if q_data.get("feedback"): st.caption(f"Feedback: {q_data["feedback"]}")
            st.markdown("---")

def display_level_box(level: int, title: str, description: str, requirements: list, user_id: int):
    with st.container(border=True):
        st.subheader(f"Level {level}: {title}")
        st.caption(description)
        
        user_progress = get_user_level_progress(user_id, level)
        is_unlocked = True
        if level > 1:
            prev_level_progress = get_user_level_progress(user_id, level - 1)
            if not prev_level_progress or prev_level_progress["status"] not in ["completed_online_tests", "eligible_for_physical_test", "fully_certified"]:
                is_unlocked = False
        
        if not is_unlocked:
            st.info(f"🔒 Locked - Complete Level {level-1} to unlock this certification level.")
            return

        if user_progress and user_progress["status"] in ["completed_online_tests", "eligible_for_physical_test", "fully_certified"]:
            st.success(f"✅ Level {level} Online Tests Completed!")
            # Certificate Handling
            existing_certificate = get_user_certificate(user_id, level)
            if existing_certificate and existing_certificate["pdf_path"] and os.path.exists(existing_certificate["pdf_path"]):
                with open(existing_certificate["pdf_path"], "rb") as fp:
                    st.download_button(
                        label=f"📜 Download Level {level} Certificate",
                        data=fp,
                        file_name=os.path.basename(existing_certificate["pdf_path"]),
                        mime="application/pdf",
                        key=f"download_cert_l{level}"
                    )
            else:
                if st.button(f"📜 Generate Level {level} Certificate", type="primary", key=f"generate_cert_l{level}"):
                    with st.spinner("Generating your certificate..."):
                        user_db_data = get_current_user_id() # Fetches full user row
                        user_full_name = user_db_data["email"].split("@")[0].replace(".", " ").title() if user_db_data else "Coach"
                        issue_date = datetime.date.today().strftime("%d/%m/%Y")
                        # Generate a unique certificate ID (can be stored in DB)
                        cert_id = f"MMA-L{level}-{user_id}-{datetime.datetime.now().timestamp()}"
                        pdf_path = generate_certificate_pdf(user_name=user_full_name, level=level, issue_date_str=issue_date, certificate_id=cert_id)
                        if pdf_path:
                            save_user_certificate(cert_id, user_id, level, pdf_path)
                            # Update progress if certificate path needs to be stored there too
                            st.success("Certificate generated! Click Download.")
                            st.rerun()
                        else:
                            st.error("Could not generate certificate at this time.")
        else:
            # Display available quizzes for this level
            quizzes_for_level = get_active_quizzes_by_level(level)
            if quizzes_for_level:
                st.markdown("**Available Quizzes:**")
                for quiz_def in quizzes_for_level:
                    user_attempts = get_user_quiz_attempts(user_id, quiz_def["id"])
                    num_attempts_taken = len(user_attempts)
                    can_attempt = num_attempts_taken < quiz_def.get("max_attempts", 1)
                    
                    # Check cooldown if applicable
                    if not can_attempt and quiz_def.get("retake_cooldown_hours") and user_attempts:
                        last_attempt_time = datetime.datetime.fromisoformat(user_attempts[0]["completed_at"])
                        cooldown_delta = datetime.timedelta(hours=quiz_def["retake_cooldown_hours"])
                        if datetime.datetime.now() > last_attempt_time + cooldown_delta:
                            can_attempt = True # Cooldown passed
                    
                    col_quiz, col_attempt = st.columns([3,1])
                    with col_quiz:
                        st.write(f"- {quiz_def["title"]}")
                        if num_attempts_taken > 0:
                            last_attempt = user_attempts[0]
                            status_emoji = "✅ Passed" if last_attempt["passed"] else "❌ Failed"
                            st.caption(f"Last attempt: {last_attempt["percentage_score"]}% ({status_emoji}). Attempts: {num_attempts_taken}/{quiz_def.get("max_attempts", 1)}")
                    with col_attempt:
                        if can_attempt:
                            if st.button(f"Start Quiz", key=f"start_quiz_{quiz_def["id"]}"):
                                questions = get_questions_for_quiz_from_db(quiz_def["id"])
                                if questions:
                                    engine = QuizEngine(user_id=user_id)
                                    engine.load_quiz(quiz_id=quiz_def["id"], quiz_definition=dict(quiz_def), questions=questions)
                                    st.session_state.active_quiz_engine = engine
                                    st.session_state.current_quiz_id = quiz_def["id"]
                                    st.session_state.quiz_submitted = False
                                    st.rerun()
                                else:
                                    st.error(f"Could not load questions for {quiz_def["title"]}. Please contact admin.")
                        else:
                            st.caption("Max attempts reached" if num_attempts_taken >= quiz_def.get("max_attempts",1) else "Cooldown active")
            else:
                st.info("No quizzes currently available for this level. Please check back later.")

        with st.expander("📋 Level Requirements"):
            for req in requirements:
                st.markdown(f"- {req}")

def certification_page():
    st.title(PAGE_TITLE)
    st.markdown("Welcome to the Coach Olympus Certification Program. This program is designed to provide structured training and certification for MMA coaches. Through a series of assessments and practical evaluations, you can earn certifications that recognize your coaching skills and knowledge.")
    st.markdown("---")

    if not is_user_logged_in():
        st.warning("Please log in to access the certification program.")
        return

    user_email = get_current_user_email()
    user_id = fetch_user_id(user_email)

    if not user_id:
        st.error("Could not retrieve user information. Please try logging out and back in.")
        return
    
    st.caption(f"Logged in as: {user_email} (User ID: {user_id})")

    # --- State Management for Quiz --- 
    if "active_quiz_engine" not in st.session_state:
        st.session_state.active_quiz_engine = None
    if "current_quiz_id" not in st.session_state:
        st.session_state.current_quiz_id = None
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    # --- Main Page Logic ---
    if st.session_state.active_quiz_engine and not st.session_state.quiz_submitted:
        display_quiz(st.session_state.active_quiz_engine, user_id)
    elif st.session_state.active_quiz_engine and st.session_state.quiz_submitted:
        display_quiz_results(st.session_state.active_quiz_engine, user_id)
    else:
        st.header("My Certification Progress")
        # Level 1
        display_level_box(
            level=1, title="MMA Coaching Fundamentals", 
            description="Master the basics of MMA coaching including safety protocols, basic techniques, and ethics.",
            requirements=["Pass the Level 1 online knowledge assessment", "Demonstrate understanding of core MMA principles", "Complete safety protocols training"],
            user_id=user_id
        )
        # Level 2
        display_level_box(
            level=2, title="Intermediate MMA Coaching", 
            description="Develop advanced coaching techniques and comprehensive training methodologies.",
            requirements=["Complete Level 1 certification", "Pass the Level 2 assessment with 75% or higher", "Submit training plan documentation", "2+ months of coaching experience"],
            user_id=user_id
        )
        # Level 3
        display_level_box(
            level=3, title="Advanced MMA Tactics & Ethics", 
            description="Master competition preparation, advanced tactics, and ethical leadership in MMA.",
            requirements=["Complete Level 2 certification", "Pass Level 3 assessment with 80% or higher", "Submit advanced coaching portfolio", "6+ months of documented coaching experience", "Endorsement from MMA Academy"],
            user_id=user_id
        )
        st.markdown("---")
        st.markdown("Coach Olympus Certification Program | © 2025 MMA Academy")

if __name__ == "__main__":
    # This page is intended to be run as part of the multipage app.
    # For standalone testing, you might need to mock st.session_state.user_info
    if "user_info" not in st.session_state:
         st.session_state.user_info = {"email": "test.user@example.com", "role": "user", "id": 1} # Mock login for testing

    # Additional DB utility imports with error handling
    try:
        from utils.db_utils import (
            init_db,
            populate_questions_from_json_files,
            add_quiz,
            link_question_to_quiz,
            get_db_connection
        )
    except ImportError as e:
        st.error(f"Failed to import database initialization utilities: {e}")
        st.stop()

    # Initialize database and populate content if needed
    if __name__ == "__main__":
        # This is usually called once at app startup or from app.py
        try:
            init_db()
            st.success("Database initialized successfully")

            # Populate questions if needed
            if st.button("Populate Questions from JSON"):
                with st.spinner("Populating questions from JSON files..."):
                    populate_questions_from_json_files()
                    st.success("Questions populated successfully")

            # Add example quizzes if needed
            if st.button("Add Example Quizzes"):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    # Example quizzes data
                    quizzes_data = [
                        {
                            "title": "Level 1 - Manual Basics Quiz (DB)",
                            "data": {
                                "level": 1,
                                "description": "Test your knowledge of MMA fundamentals.",
                                "source_topic_file": "manual_basics_l1.json",
                                "num_questions": 3,
                                "passing_score_percentage": 70.0,
                                "max_attempts": 2,
                                "time_limit_minutes": 10,
                                "retake_cooldown_hours": 1,
                                "is_active": 1
                            },
                            "questions": ["L1MB001", "L1MB002", "L1MB005"]
                        },
                        {
                            "title": "Level 2 - GAMMA Regulations Quiz (DB)",
                            "data": {
                                "level": 2,
                                "description": "Test your knowledge of GAMMA rules.",
                                "source_topic_file": "gamma_full_regulations_l2.json",
                                "num_questions": 2,
                                "passing_score_percentage": 75.0,
                                "max_attempts": 1,
                                "time_limit_minutes": 15,
                                "retake_cooldown_hours": None,
                                "is_active": 1
                            },
                            "questions": ["L2GR001", "L2GR002"]
                        }
                    ]

                    for quiz in quizzes_data:
                        # Check if quiz exists
                        cursor.execute("SELECT 1 FROM Quizzes WHERE title = ?", (quiz["title"],))
                        if not cursor.fetchone():
                            # Add quiz data
                            quiz_data = quiz["data"]
                            quiz_data["title"] = quiz["title"]
                            quiz_id = add_quiz(quiz_data)

                            if quiz_id:
                                # Link questions
                                for question_id in quiz["questions"]:
                                    link_question_to_quiz(quiz_id, question_id)
                                st.success(f"Added quiz: {quiz['title']}")
                            else:
                                st.warning(f"Failed to add quiz: {quiz['title']}")

                    conn.close()
                    st.success("Example quizzes added successfully")

                except Exception as e:
                    st.error(f"Error adding example quizzes: {e}")
                    if 'conn' in locals():
                        conn.close()

        except Exception as e:
            st.error(f"Error during initialization: {e}")

        # Proceed with certification page
        certification_page()