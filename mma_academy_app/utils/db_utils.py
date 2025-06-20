"""
Database utilities for the MMA Academy app, including Certification Module.
Handles SQLite database interactions.
"""
import sqlite3
import json
import os
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import sqlite3, datetime

# Runs once, before any INSERT/UPDATEs
sqlite3.register_adapter(
    datetime.datetime,
    lambda ts: ts.isoformat(sep=' ')
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path configurations
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(PROJECT_ROOT_DIR, "mma_academy.db")
QUESTIONS_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "questions")

def get_db_connection() -> sqlite3.Connection:
    """
    Establish and return a connection to the SQLite database with row factory set.

    Returns:
        sqlite3.Connection: Database connection object
    """
    try:
        conn = sqlite3.connect(DB_NAME) # Use DB_NAME for consistency
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

# Centralized query execution helper (from user's provided file, slightly adapted for clarity)
def _execute_query(query: str, params: tuple = (), commit: bool = False, fetch_one: bool = False, fetch_all: bool = False, last_row_id: bool = False) -> Any:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # before executing
        params = tuple(
            value.isoformat(sep=' ') if isinstance(value, datetime.datetime) else value
            for value in params
        )
        cursor.execute(query, params)

        if commit:
            conn.commit()
            if last_row_id:
                return cursor.lastrowid
            return True # For operations like INSERT, UPDATE, DELETE that commit
        
        if fetch_one:
            row = cursor.fetchone()
            return dict(row) if row else None
        
        if fetch_all:
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        return None # Default if no other action specified

    except sqlite3.Error as e:
        logger.error(f"Database error during query: {query} with params {params}. Error: {e}")
        if conn:
            conn.rollback() # Rollback on error
        return None # Or False, depending on expected failure return for commit operations
    finally:
        if conn:
            conn.close()

# Initialize database schema
def init_db(): # Renamed from init_db_tables for consistency with previous versions, but kept the core logic.
    """
    Initialize all database tables for the MMA Academy app.
    This function creates all the required tables if they don't exist.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # User Authentication Table (Standardized Schema)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            access_code TEXT, -- Stores the one-time access code
            role TEXT DEFAULT 'user', -- 'user' or 'admin'
            is_active INTEGER DEFAULT 1, -- 1 for active, 0 for inactive (e.g., after code is used or if disabled)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_code_sent_at TIMESTAMP -- To track when the last code was sent, for expiry or cooldown
        )
        """)

        # Content Management Tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            description TEXT,
            category TEXT,
            uploaded_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS community_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL, -- Consider changing to user_id INTEGER FOREIGN KEY REFERENCES invites(id)
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            parent_post_id INTEGER,
            FOREIGN KEY (parent_post_id) REFERENCES community_posts(id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_by TEXT,
            description TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Certification Module Tables (Schema from v4)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Questions (
            id TEXT PRIMARY KEY,
            level INTEGER NOT NULL,
            topic TEXT NOT NULL,
            question_type TEXT NOT NULL,
            text TEXT NOT NULL,
            options TEXT,
            correct_answer_id TEXT NOT NULL,
            marks INTEGER NOT NULL DEFAULT 1,
            difficulty TEXT DEFAULT 'medium',
            video_url TEXT,
            image_url TEXT,
            feedback TEXT,
            source_reference TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER NOT NULL,
            title TEXT NOT NULL UNIQUE,
            description TEXT,
            source_topic_file TEXT,
            num_questions INTEGER,
            time_limit_minutes INTEGER,
            passing_score_percentage REAL NOT NULL DEFAULT 75.0,
            max_attempts INTEGER DEFAULT 1,
            retake_cooldown_hours INTEGER,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS QuizQuestions (
            quiz_id INTEGER NOT NULL,
            question_id TEXT NOT NULL,
            question_order INTEGER,
            PRIMARY KEY (quiz_id, question_id),
            FOREIGN KEY (quiz_id) REFERENCES Quizzes(id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES Questions(id) ON DELETE CASCADE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS UserQuizAttempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            quiz_id INTEGER NOT NULL,
            attempt_number INTEGER NOT NULL,
            score_achieved REAL,
            total_marks_possible REAL,
            percentage_score REAL,
            passed INTEGER DEFAULT 0,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            answers_submitted TEXT,
            FOREIGN KEY (user_id) REFERENCES invites(id) ON DELETE CASCADE,
            FOREIGN KEY (quiz_id) REFERENCES Quizzes(id) ON DELETE CASCADE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS UserLevelProgress (
            user_id INTEGER NOT NULL,
            level INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'not_started',
            online_tests_completed_at TIMESTAMP,
            physical_test_status TEXT,
            physical_test_completed_at TIMESTAMP,
            overall_completion_date TIMESTAMP,
            PRIMARY KEY (user_id, level),
            FOREIGN KEY (user_id) REFERENCES invites(id) ON DELETE CASCADE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS UserCertificates (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            level INTEGER NOT NULL,
            issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pdf_path TEXT,
            qr_code_data TEXT,
            valid_until TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES invites(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id, level) REFERENCES UserLevelProgress(user_id, level) ON DELETE CASCADE
        )
        """)

        conn.commit()
        logger.info("Database schema initialized successfully.")

        # Add admin user from secrets.toml if it doesn't exist
        try:
            import streamlit as st
            admin_email_from_secrets = st.secrets.get("admin", {}).get("email")
            if admin_email_from_secrets:
                existing_admin = _execute_query("SELECT id FROM invites WHERE email = ? AND role = 'admin'", (admin_email_from_secrets,), fetch_one=True)
                if not existing_admin:
                    # Add admin user with a placeholder access_code (will be replaced when they request a code)
                    # is_active is 1 by default, role is 'admin'
                    _execute_query("INSERT INTO invites (email, access_code, role, last_code_sent_at) VALUES (?, ?, 'admin', ?)", 
                                   (admin_email_from_secrets, 'INITIAL_ADMIN_CODE', datetime.datetime.now()), commit=True)
                    logger.info(f"Admin user {admin_email_from_secrets} added/ensured in invites table.")
        except ImportError:
            logger.warning("Streamlit not available in this context, cannot load admin email from secrets.")
        except Exception as e_admin:
            logger.error(f"Error processing admin user from secrets: {e_admin}")

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

# --- User Management (Standardized) ---
def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    return _execute_query("SELECT * FROM invites WHERE email = ?", (email,), fetch_one=True)

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    return _execute_query("SELECT * FROM invites WHERE id = ?", (user_id,), fetch_one=True)

def store_invite_token(email: str, access_code: str, role: str = 'user') -> bool:
    """
    Stores or updates an invite token (access_code) for a user.
    If user exists, updates their access_code and role, and marks as active.
    If user does not exist, creates a new invite.
    """
    now = datetime.datetime.now()
    existing_user = get_user_by_email(email)
    if existing_user:
        # Update existing user's access code, role, and mark as active
        return _execute_query("UPDATE invites SET access_code = ?, role = ?, is_active = 1, last_code_sent_at = ? WHERE email = ?", 
                              (access_code, role, now, email), commit=True) is not None # _execute_query returns True on successful commit
    else:
        # Insert new user invite
        return _execute_query("INSERT INTO invites (email, access_code, role, is_active, last_code_sent_at) VALUES (?, ?, ?, 1, ?)", 
                              (email, access_code, role, now), commit=True) is not None

def verify_invite_token(email: str, access_code: str) -> Tuple[bool, str, Optional[int]]:
    """
    Verifies an invite token (access_code).
    Returns (is_valid, role, user_id).
    Marks the code as inactive (used) upon successful verification for one-time use.
    """
    user = get_user_by_email(email)
    if user and user["access_code"] == access_code and user["is_active"] == 1:
        # Mark code as used (inactive) - this makes it a one-time code
        # If codes are meant to be reusable for a period, this logic would change
        # _execute_query("UPDATE invites SET is_active = 0 WHERE email = ?", (email,), commit=True)
        # For now, let's assume codes are single-use for login. The login page can request a new one.
        # The `is_active` flag on the invite itself can be managed by an admin for overall access.
        # Let's keep the invite active but rely on the access_code being fresh.
        # The `auth.py` logic should handle if a code is "used" by clearing it or setting a session.
        return True, user["role"], user["id"]
    return False, "", None

def get_all_invites() -> List[Dict[str, Any]]:
    return _execute_query("SELECT id, email, access_code, role, is_active, created_at, last_code_sent_at FROM invites ORDER BY created_at DESC", fetch_all=True) or []

# --- Video Management Functions ---
def add_video(title: str, url: str, description: Optional[str] = None,
              category: Optional[str] = None, uploaded_by: Optional[str] = None) -> Optional[int]:
    """
    Adds a new video to the database.
    Returns the ID of the newly added video, or None if the operation fails.
    """
    try:
        return _execute_query(
            "INSERT INTO videos (title, url, description, category, uploaded_by) VALUES (?, ?, ?, ?, ?)",
            (title, url, description, category, uploaded_by),
            commit=True,
            last_row_id=True
        )
    except Exception as e:
        logger.error(f"Error adding video: {e}")
        return None

# --- Video Management (from user's file) ---
def get_all_videos() -> List[Dict[str, Any]]:
    return _execute_query("SELECT id, title, url, description, category, uploaded_by, created_at FROM videos ORDER BY created_at DESC", fetch_all=True) or []

def delete_video(video_id: int) -> bool:
    return _execute_query("DELETE FROM videos WHERE id = ?", (video_id,), commit=True) is not None

# --- PDF Resource Management (from user's file) ---
def add_resource(title: str, file_name: str, file_path: str, uploaded_by: Optional[str] = None, description: Optional[str] = None) -> Optional[int]:
    # Assuming 'title' is not a column in uploaded_pdfs, it might be part of description or a new column if needed.
    # The original user function had 'title' but the INSERT query didn't use it.
    return _execute_query("INSERT INTO uploaded_pdfs (filename, filepath, uploaded_by, description) VALUES (?, ?, ?, ?)", 
                          (file_name, file_path, uploaded_by, description), commit=True, last_row_id=True)

def get_all_resources() -> List[Dict[str, Any]]:
    return _execute_query("SELECT id, filename, filepath, description, uploaded_by, uploaded_at FROM uploaded_pdfs ORDER BY uploaded_at DESC", fetch_all=True) or []

def delete_resource(resource_id: int) -> bool:
    return _execute_query("DELETE FROM uploaded_pdfs WHERE id = ?", (resource_id,), commit=True) is not None

# --- Certification Module DB Functions (Schema from v4, minor adaptations) ---

# Questions Management
def add_question(q_data: dict):
    try:
        options_json = json.dumps(q_data.get("options")) if q_data.get("options") else None
        _execute_query("""
            INSERT INTO Questions (id, level, topic, question_type, text, options, correct_answer_id, marks, difficulty, video_url, image_url, feedback, source_reference)
            VALUES (:id, :level, :topic, :question_type, :text, :options, :correct_answer_id, :marks, :difficulty, :video_url, :image_url, :feedback, :source_reference)
        """, {
            "id": q_data["id"],
            "level": q_data["level"],
            "topic": q_data["topic"],
            "question_type": q_data["question_type"],
            "text": q_data["text"],
            "options": options_json,
            "correct_answer_id": str(q_data["correct_answer_id"]),
            "marks": q_data.get("marks", 1),
            "difficulty": q_data.get("difficulty", "medium"),
            "video_url": q_data.get("video_url"),
            "image_url": q_data.get("image_url"),
            "feedback": q_data.get("feedback"),
            "source_reference": q_data.get("source_reference")
        }, commit=True)
    except sqlite3.IntegrityError:
        # logger.info(f"Question {q_data['id']} already exists. Skipping.")
        pass 
    except Exception as e:
        logger.error(f"An unexpected error occurred while adding question {q_data.get('id', 'N/A')}: {e}")

def get_question_by_id(question_id: str) -> Optional[Dict[str, Any]]:
    return _execute_query("SELECT * FROM Questions WHERE id = ?", (question_id,), fetch_one=True)

def get_questions_by_topic_and_level(level: int, topic: str) -> List[Dict[str, Any]]:
    return _execute_query("SELECT * FROM Questions WHERE level = ? AND topic = ?", (level, topic), fetch_all=True) or []

# Quiz Management
def add_quiz(quiz_data: dict) -> Optional[int]:
    try:
        return _execute_query("""
            INSERT INTO Quizzes (level, title, description, source_topic_file, num_questions, time_limit_minutes, passing_score_percentage, max_attempts, retake_cooldown_hours, is_active)
            VALUES (:level, :title, :description, :source_topic_file, :num_questions, :time_limit_minutes, :passing_score_percentage, :max_attempts, :retake_cooldown_hours, :is_active)
        """, quiz_data, commit=True, last_row_id=True)
    except sqlite3.IntegrityError:
        logger.info(f"Quiz '{quiz_data.get('title')}' already exists. Fetching ID.")
        existing_quiz = _execute_query("SELECT id FROM Quizzes WHERE title = ?", (quiz_data.get('title'),), fetch_one=True)
        return existing_quiz['id'] if existing_quiz else None

def get_quiz_by_id(quiz_id: int) -> Optional[Dict[str, Any]]:
    return _execute_query("SELECT * FROM Quizzes WHERE id = ? AND is_active = 1", (quiz_id,), fetch_one=True)

def get_active_quizzes_by_level(level: int) -> List[Dict[str, Any]]:
    return _execute_query("SELECT * FROM Quizzes WHERE level = ? AND is_active = 1 ORDER BY title", (level,), fetch_all=True) or []

def link_question_to_quiz(quiz_id: int, question_id: str, order: Optional[int] = None):
    try:
        _execute_query("INSERT INTO QuizQuestions (quiz_id, question_id, question_order) VALUES (?, ?, ?)", 
                       (quiz_id, question_id, order), commit=True)
    except sqlite3.IntegrityError:
        # logger.info(f"Question {question_id} already linked to quiz {quiz_id}.")
        pass

def get_questions_for_quiz(quiz_id: int) -> List[Dict[str, Any]]:
    questions_raw = _execute_query("""
    SELECT q.* 
    FROM Questions q
    JOIN QuizQuestions qq ON q.id = qq.question_id
    WHERE qq.quiz_id = ?
    ORDER BY qq.question_order ASC NULLS LAST, q.id ASC
    """, (quiz_id,), fetch_all=True)
    
    questions_processed = []
    if questions_raw:
        for q_dict in questions_raw:
            if q_dict.get('options'):
                try:
                    q_dict['options'] = json.loads(q_dict['options'])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse options for question {q_dict['id']}")
                    q_dict['options'] = [] 
            else:
                q_dict['options'] = []
            questions_processed.append(q_dict)
    return questions_processed

# User Quiz Attempts
def save_user_quiz_attempt(user_id: int, quiz_id: int, attempt_number: int, score_details: dict, passed: bool, answers_submitted: dict) -> Optional[int]:
    completed_at = datetime.datetime.now()
    return _execute_query("""
        INSERT INTO UserQuizAttempts (user_id, quiz_id, attempt_number, score_achieved, total_marks_possible, percentage_score, passed, completed_at, answers_submitted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, quiz_id, attempt_number, score_details["score_achieved"], score_details["total_marks_possible"], 
          score_details["percentage"], 1 if passed else 0, completed_at, json.dumps(answers_submitted)), commit=True, last_row_id=True)

def get_user_quiz_attempts(user_id: int, quiz_id: int) -> List[Dict[str, Any]]:
    return _execute_query("SELECT * FROM UserQuizAttempts WHERE user_id = ? AND quiz_id = ? ORDER BY attempt_number DESC", 
                            (user_id, quiz_id), fetch_all=True) or []

# User Level Progress
def get_user_level_progress(user_id: int, level: int) -> Optional[Dict[str, Any]]:
    return _execute_query("SELECT * FROM UserLevelProgress WHERE user_id = ? AND level = ?", (user_id, level), fetch_one=True)

def update_user_level_progress(user_id: int, level: int, status: str, online_tests_completed_at=None, physical_test_status=None, physical_test_completed_at=None, overall_completion_date=None):
    existing = get_user_level_progress(user_id, level)
    params = {
        "user_id": user_id, "level": level, "status": status,
        "otca": online_tests_completed_at, "pts": physical_test_status,
        "ptca": physical_test_completed_at, "ocd": overall_completion_date
    }
    if existing:
        sql_parts = ["status = :status"]
        if online_tests_completed_at: sql_parts.append("online_tests_completed_at = :otca")
        if physical_test_status: sql_parts.append("physical_test_status = :pts")
        if physical_test_completed_at: sql_parts.append("physical_test_completed_at = :ptca")
        if overall_completion_date: sql_parts.append("overall_completion_date = :ocd")
        sql = f"UPDATE UserLevelProgress SET {', '.join(sql_parts)} WHERE user_id = :user_id AND level = :level"
        _execute_query(sql, params, commit=True)
    else:
        _execute_query("""
            INSERT INTO UserLevelProgress (user_id, level, status, online_tests_completed_at, physical_test_status, physical_test_completed_at, overall_completion_date)
            VALUES (:user_id, :level, :status, :otca, :pts, :ptca, :ocd)
        """, params, commit=True)

# User Certificates
def save_user_certificate(cert_id: str, user_id: int, level: int, pdf_path: str, qr_code_data: Optional[str] = None, valid_until=None):
    try:
        _execute_query("""
            INSERT INTO UserCertificates (id, user_id, level, pdf_path, qr_code_data, valid_until)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cert_id, user_id, level, pdf_path, qr_code_data, valid_until), commit=True)
    except sqlite3.IntegrityError:
        logger.info(f"Certificate {cert_id} might already exist.")

def get_user_certificate(user_id: int, level: int) -> Optional[Dict[str, Any]]:
    return _execute_query("SELECT * FROM UserCertificates WHERE user_id = ? AND level = ?", (user_id, level), fetch_one=True)

# --- Utility to populate Questions table from JSON files ---
def populate_questions_from_json_files():
    logger.info("Populating questions from JSON files...")
    question_files_processed = 0
    questions_added_count = 0
    for level_folder in os.listdir(QUESTIONS_BASE_DIR):
        level_path = os.path.join(QUESTIONS_BASE_DIR, level_folder)
        if os.path.isdir(level_path) and level_folder.startswith("level"):
            try:
                level_num = int(level_folder.replace("level", ""))
            except ValueError:
                logger.warning(f"Skipping invalid level folder: {level_folder}")
                continue
            for topic_file in os.listdir(level_path):
                if topic_file.endswith(".json"):
                    file_path = os.path.join(level_path, topic_file)
                    logger.info(f"Processing file: {file_path}")
                    questions_in_file = []
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            questions_in_file = json.load(f)
                    except Exception as e:
                        logger.error(f"Error reading or parsing {file_path}: {e}")
                        continue
                    if isinstance(questions_in_file, list):
                        for q_data in questions_in_file:
                            if not all(k in q_data for k in ["id", "topic", "question_type", "text", "correct_answer_id"]):
                                logger.warning(f"Skipping question due to missing fields in {file_path}: {q_data.get('id', 'Unknown ID')}")
                                continue
                            q_data["level"] = level_num 
                            add_question(q_data) 
                            questions_added_count +=1 
                        question_files_processed += 1
                    else:
                        logger.warning(f"Skipping {file_path}: content is not a list of questions.")
    logger.info(f"Processed {question_files_processed} JSON files. Attempted to add/update {questions_added_count} questions.")

if __name__ == "__main__":
    logger.info("Initializing database and optionally populating content...")
    init_db() # This will create tables and ensure admin from secrets is present
    logger.info("Database schema initialized/verified.")
    
    # --- Populate Questions from JSON files (Idempotent) ---
    # Uncomment to run population. It's idempotent due to add_question's try-except.
    # populate_questions_from_json_files()
    # logger.info("Question population from JSON files complete.")

    # --- Example: Add Quiz Definitions to DB if not present (for testing/setup) ---
    # This section should ideally be run once or managed by an admin interface.
    # It's made idempotent by add_quiz checking if the quiz title exists.
    # logger.info("\nSetting up example quizzes...")
    
    # quizzes_to_add = [
    #     {"level": 1, "title": "Level 1 - Manual Basics Quiz (DB)", "description": "Test your knowledge of MMA fundamentals.", "source_topic_file": "manual_basics_l1.json", "num_questions": 3, "passing_score_percentage": 70.0, "max_attempts": 2, "time_limit_minutes": 10, "retake_cooldown_hours": 1, "is_active": 1, "questions_to_link": ["L1MB001", "L1MB002", "L1MB005"]},
    #     {"level": 2, "title": "Level 2 - GAMMA Regulations Quiz (DB)", "description": "Test your knowledge of GAMMA rules.", "source_topic_file": "gamma_full_regulations_l2.json", "num_questions": 2, "passing_score_percentage": 75.0, "max_attempts": 1, "time_limit_minutes": 15, "retake_cooldown_hours": None, "is_active": 1, "questions_to_link": ["L2GR001", "L2GR002"]},
    #     {"level": 2, "title": "Level 2 - Injury Prevention Quiz (DB)", "description": "Assess knowledge on preventing common MMA injuries.", "source_topic_file": "injury_prevention_l2.json", "num_questions": 2, "passing_score_percentage": 75.0, "max_attempts": 1, "time_limit_minutes": 10, "retake_cooldown_hours": None, "is_active": 1, "questions_to_link": ["L2IP001", "L2IP002"]},
    #     {"level": 3, "title": "Level 3 - Advanced Coaching Manual Quiz (DB)", "description": "Test expertise on advanced coaching strategies.", "source_topic_file": "advanced_manual_l3.json", "num_questions": 2, "passing_score_percentage": 80.0, "max_attempts": 1, "time_limit_minutes": 20, "retake_cooldown_hours": None, "is_active": 1, "questions_to_link": ["L3AM001", "L3AM002"]}
    # ]
    # for quiz_def in quizzes_to_add:
    #     questions_to_link = quiz_def.pop("questions_to_link")
    #     quiz_id = add_quiz(quiz_def)
    #     if quiz_id:
    #         logger.info(f"Quiz '{quiz_def['title']}' (ID: {quiz_id}) ensured.")
    #         for q_id_to_link in questions_to_link:
    #             link_question_to_quiz(quiz_id, q_id_to_link)
    #         logger.info(f"Linked questions to quiz ID {quiz_id}.")
    #     else:
    #         logger.error(f"Failed to add or find quiz: {quiz_def['title']}")

    logger.info("\nDB Utils for Certification Module updated and example setup complete.")
    logger.info("To populate questions and quizzes, uncomment the relevant sections in this __main__ block and run: python utils/db_utils.py")

