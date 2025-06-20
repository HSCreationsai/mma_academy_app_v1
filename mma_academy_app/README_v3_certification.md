# Indonesia MMA Youth Excellence Academy & National Coach Training Program - Streamlit App (v3 - Coach Certification Module)

This application, featuring **Coach Olympus**, serves as an intelligent assistant and training portal for MMA coaches and certified athletes associated with the Indonesia MMA Youth Excellence Academy & National Coach Training Program. This version introduces the initial implementation of the **Coach Certification Module**.

## Features (Including New Certification Module - Phase 1)

*   **Invite-Only Login:** Secure email-token based authentication.
*   **Dashboard:** Personalized welcome and (placeholder) progress summary.
*   **Video Center:** Stream admin-curated training videos.
*   **Manual Reader:** Read the comprehensive PDF training manual page by page, with Text-to-Speech (TTS) support.
*   **Coach Olympus (RAG Chatbot):** Interact with an AI coach powered by a local LLM.
*   **Community Wall:** Internal discussion forum for members.
*   **Admin Panel:** For administrators to manage video content, upload PDF resources, and manage user invites. (Future enhancements will include managing quizzes and questions).
*   **NEW: Coach Certification Module (Page: `8_Coach_Certification.py`):**
    *   **Level 1 Quiz:** Users can now take a Level 1 quiz based on questions from `questions/level1/manual_basics_l1.json`.
    *   **Quiz Engine:** A foundational quiz engine (`utils/quiz_engine.py`) manages quiz loading, state, and scoring.
    *   **Level 1 Certificate Generation:** Upon successfully passing the Level 1 quiz, users can generate and download a PDF certificate for Level 1 (`utils/certificate_generator.py` with templates in `templates/certificates/`).
    *   **Database Integration:** New database tables (`Questions`, `Quizzes`, `UserQuizAttempts`, `UserLevelProgress`, `UserCertificates`) are created by `utils/db_utils.py` to store certification data. The Level 1 quiz page currently uses mock DB interactions for simplicity in this initial rollout but the schema is ready.
    *   **Placeholders for Level 2 & 3:** The UI shows locked placeholders for future Level 2 and 3 certifications.

## Key Files and Directories for Certification Module

*   `pages/8_Coach_Certification.py`: The main Streamlit page for users to interact with the certification program.
*   `utils/quiz_engine.py`: Core logic for loading and managing quizzes.
*   `utils/certificate_generator.py`: Handles PDF certificate generation using WeasyPrint.
*   `utils/db_utils.py`: Updated with new tables and functions for certification data (though the Streamlit page uses mock data for now).
*   `questions/level1/manual_basics_l1.json`: Sample questions for the Level 1 Manual Basics quiz.
*   `templates/certificates/level1_certificate_template.html`: HTML template for the Level 1 certificate.
*   `templates/certificates/certificate_styles.css`: CSS for styling certificates.
*   `user_certificates/`: Directory where generated PDF certificates will be saved.

## Setup and Installation (Windows with PyCharm & Python 3.12.x)

Follow the instructions in `README_v2.md` for the general application setup. Key points for the new module:

1.  **Extract `mma_academy_app_v3_certification.zip`**.
2.  **Install Dependencies:** Ensure all dependencies from `requirements.txt` are installed. This includes `weasyprint` for certificate generation.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: WeasyPrint might have system-level dependencies (like Pango, Cairo, GDK-PixBuf). If you encounter issues installing or running WeasyPrint on Windows, refer to its official installation guide for Windows.* 
3.  **Database Initialization:** When you first run the application, or if you run `python utils/db_utils.py` directly, the `init_db()` function will attempt to create all necessary tables, including the new ones for the certification module.
4.  **Populate Questions (Manual Step for now, or via Admin Panel in future):
    *   The `manual_basics_l1.json` is provided. To load these into the database, you would typically call a function like `populate_questions_from_json_files()` from `db_utils.py` (e.g., by uncommenting it in the `if __name__ == "__main__":` block of `db_utils.py` and running the script `python utils/db_utils.py`).
    *   Similarly, an admin would define the "Level 1 - Manual Basics Quiz" in the `Quizzes` table and link it to these questions. The current `8_Coach_Certification.py` uses a mock quiz definition for simplicity in this first version.

## Running the Application

1.  Navigate to the project root directory (e.g., `D:\mma_academy_app>`).
2.  Run: `streamlit run mma_academy_app/app.py`
3.  Access the "Coach Certification" page from the sidebar.

## Next Steps for Full Certification Module Functionality

*   **Full DB Integration in Streamlit Page:** Replace mock DB calls in `8_Coach_Certification.py` with actual calls to `db_utils.py` functions for loading quiz definitions, questions, saving attempts, and tracking progress.
*   **Admin Panel Enhancements:** Build UI in the Admin Panel for:
    *   Managing questions (uploading JSONs, viewing, editing).
    *   Defining quizzes in the `Quizzes` table and linking questions.
    *   Viewing user certification progress and quiz results.
*   **Content Population:** Create and populate JSON question files for all topics and levels (Level 1, 2, 3).
*   **Certificate Templates:** Design and implement HTML/CSS templates for Level 2 and Level 3 certificates.
*   **Progression Logic:** Fully implement the locking/unlocking of levels based on database-tracked progress.

This version provides the foundational elements for the Level 1 certification process. Enjoy exploring Coach Olympus and the new certification features!
