# Indonesia MMA Youth Excellence Academy & National Coach Training Program - Streamlit App (v5 - DB Fix & Full Coach Certification Module)

This application, featuring **Coach Olympus**, serves as an intelligent assistant and training portal for MMA coaches and certified athletes associated with the Indonesia MMA Youth Excellence Academy & National Coach Training Program. This version includes the **Full Coach Certification Module for Levels 1, 2, and 3**, and incorporates critical fixes for database schema and authentication logic.

## Key Changes in This Version (v5_db_fix)

*   **Database Schema Correction:** The `invites` table schema within `utils/db_utils.py` has been standardized and corrected to resolve login and token storage errors. Related authentication functions have been updated for consistency.
*   **Full Certification Module:** Retains all features from v4, including multi-level quizzes (Levels 1, 2, 3), certificate generation for all levels, and database integration for tracking progress.

## Features

*   **Invite-Only Login:** Secure email-token based authentication (now with corrected database logic).
*   **Dashboard:** Personalized welcome and progress summary.
*   **Video Center:** Stream admin-curated training videos.
*   **Manual Reader:** Read the comprehensive PDF training manual page by page, with Text-to-Speech (TTS) support.
*   **Coach Olympus (RAG Chatbot):** Interact with an AI coach powered by a local LLM.
*   **Community Wall:** Internal discussion forum for members.
*   **Admin Panel:** For administrators to manage video content, upload PDF resources, and manage user invites.
*   **FULL Coach Certification Module (Page: `8_Coach_Certification.py`):**
    *   Multi-Level Quizzes (Levels 1, 2, 3).
    *   Quiz Engine (`utils/quiz_engine.py`).
    *   Certificate Generation for Levels 1, 2, 3 (`utils/certificate_generator.py`, HTML/CSS templates).
    *   Full Database Integration (`utils/db_utils.py`) for questions, quizzes, attempts, progress, and certificates.
    *   User Progression UI reflecting all three levels.

## Key Files and Directories (Unchanged from v4 for Certification Content)

*   `pages/8_Coach_Certification.py`
*   `utils/quiz_engine.py`, `utils/certificate_generator.py`
*   `utils/db_utils.py` (This file is the primary one updated in v5 for the fix)
*   `questions/level1/`, `questions/level2/`, `questions/level3/`
*   `templates/certificates/`
*   `user_certificates/`

## CRITICAL: Setup and Installation (Windows with PyCharm & Python 3.12.x)

**If you are upgrading from a previous version and encountered the login/database error, follow these steps carefully:**

1.  **Extract `mma_academy_app_v5_db_fix.zip`** to a new clean directory, or ensure you replace all relevant files in your existing project, especially `utils/db_utils.py`.
2.  **DELETE YOUR EXISTING `mma_academy.db` FILE:** This file is typically located in your project root (e.g., `D:\mma_academy_app\mma_academy_app\` or `D:\mma_academy_app\`). This step is **ESSENTIAL** to allow the corrected schema to be created.
3.  **Install/Verify Dependencies:** Ensure all dependencies from `requirements.txt` are installed in your virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    Ensure `weasyprint` and its system dependencies (Pango, Cairo, GDK-PixBuf for Windows) are correctly installed.
4.  **Initialize New Database & Populate Content:**
    *   Open your terminal/command prompt in the project root directory (where `app.py` is located).
    *   Activate your Python virtual environment.
    *   Run the `db_utils.py` script **directly once** to create the new database with the corrected schema and populate initial data:
        ```bash
        python utils/db_utils.py
        ```
        This script will:
        1.  Call `init_db()`: Creates all tables with the correct schema, including the fixed `invites` table. It will also attempt to add the admin user specified in your `.streamlit/secrets.toml` file.
        2.  The `if __name__ == "__main__":` block in `db_utils.py` contains commented-out sections for `populate_questions_from_json_files()` and for adding quiz definitions (e.g., `quizzes_to_add`). **You MUST uncomment these sections (or at least the parts you need) and run `python utils/db_utils.py` again** to load all questions into the `Questions` table and to define the quizzes in the `Quizzes` table and link them. Without this, the certification page will not find any quizzes.

## Running the Application

1.  Navigate to the project root directory.
2.  Ensure your virtual environment is activated.
3.  Run: `streamlit run app.py` (if `app.py` is in the root of `mma_academy_app`) or `streamlit run mma_academy_app/app.py` (if you are one level above).
4.  Attempt to log in. The admin email from your `secrets.toml` should now be correctly recognized after the database re-initialization. You will need to request an access code for it first via the "Request Access Code" tab on the login page.
5.  Access the "Coach Certification" page to test the multi-level quizzes.

This version aims to resolve the critical database and login issues while retaining all the advanced certification module functionality. Please follow the database reset instructions carefully.
