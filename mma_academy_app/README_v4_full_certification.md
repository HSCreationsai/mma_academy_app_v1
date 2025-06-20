# Indonesia MMA Youth Excellence Academy & National Coach Training Program - Streamlit App (v4 - Full Coach Certification Module)

This application, featuring **Coach Olympus**, serves as an intelligent assistant and training portal for MMA coaches and certified athletes associated with the Indonesia MMA Youth Excellence Academy & National Coach Training Program. This version includes the **Full Coach Certification Module for Levels 1, 2, and 3**.

## Features (Including Full Certification Module)

*   **Invite-Only Login:** Secure email-token based authentication.
*   **Dashboard:** Personalized welcome and progress summary.
*   **Video Center:** Stream admin-curated training videos.
*   **Manual Reader:** Read the comprehensive PDF training manual page by page, with Text-to-Speech (TTS) support.
*   **Coach Olympus (RAG Chatbot):** Interact with an AI coach powered by a local LLM.
*   **Community Wall:** Internal discussion forum for members.
*   **Admin Panel:** For administrators to manage video content, upload PDF resources, and manage user invites. (Future enhancements will include managing quizzes, questions, and user certification progress).
*   **FULL Coach Certification Module (Page: `8_Coach_Certification.py`):**
    *   **Multi-Level Quizzes:** Users can now access and take quizzes for Level 1, Level 2, and Level 3.
        *   **Level 1 Questions:** Based on `questions/level1/manual_basics_l1.json`.
        *   **Level 2 Questions:** Based on `questions/level2/gamma_full_regulations_l2.json` and `questions/level2/injury_prevention_l2.json`.
        *   **Level 3 Questions:** Based on `questions/level3/advanced_manual_l3.json`.
    *   **Quiz Engine (`utils/quiz_engine.py`):** Manages quiz loading, state, and scoring for all levels.
    *   **Certificate Generation (Levels 1, 2, 3):** Upon successfully passing quizzes for each level, users can generate and download PDF certificates.
        *   Templates: `templates/certificates/level1_certificate_template.html`, `level2_certificate_template.html`, `level3_certificate_template.html`.
        *   Styles: `templates/certificates/certificate_styles.css`.
        *   Logic: `utils/certificate_generator.py` (updated for all levels).
    *   **Database Integration (`utils/db_utils.py`):** Full database support for storing questions, quiz definitions, user attempts, user progress across all levels, and certificate records. The Streamlit page now uses these DB functions for a persistent experience.
    *   **User Progression:** The UI now reflects progress through Levels 1, 2, and 3, with appropriate locking/unlocking logic based on completion of previous levels.

## Key Files and Directories for Full Certification Module

*   `pages/8_Coach_Certification.py`: The main Streamlit page, now supporting all three certification levels.
*   `utils/quiz_engine.py`: Core logic for loading and managing quizzes.
*   `utils/certificate_generator.py`: Handles PDF certificate generation for Levels 1, 2, and 3.
*   `utils/db_utils.py`: Fully updated with tables and functions for all certification data.
*   `questions/level1/`, `questions/level2/`, `questions/level3/`: Contain JSON question files for each respective level.
*   `templates/certificates/`: Contains HTML templates and CSS for all certificate levels.
*   `user_certificates/`: Directory where generated PDF certificates will be saved.

## Setup and Installation (Windows with PyCharm & Python 3.12.x)

Follow the instructions in previous READMEs (`README_v2.md`, `README_v3_certification.md`) for general application setup. Key points for this full version:

1.  **Extract `mma_academy_app_v4_full_certification.zip`**.
2.  **Install Dependencies:** Ensure all dependencies from `requirements.txt` are installed. This includes `weasyprint`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: WeasyPrint might have system-level dependencies (like Pango, Cairo, GDK-PixBuf). If you encounter issues installing or running WeasyPrint on Windows, refer to its official installation guide for Windows. The user mentioned GTK is installed, which should help.*
3.  **Database Initialization & Population:**
    *   Run `python utils/db_utils.py` directly **ONCE** from your project root in your activated virtual environment. This script will:
        1.  Call `init_db()` to create all necessary tables (idempotent).
        2.  Call `populate_questions_from_json_files()` to load all questions from the `questions/level1/`, `questions/level2/`, and `questions/level3/` directories into the `Questions` table.
        3.  The `if __name__ == "__main__":` block in `db_utils.py` now includes example code (commented out by default) to add quiz definitions to the `Quizzes` table and link them to the populated questions. **You should uncomment and adapt these sections in `db_utils.py` and run it once to set up the quizzes in the database.** This step is crucial for the certification page to find and load the quizzes.

## Running the Application

1.  Navigate to the project root directory (e.g., `D:\mma_academy_app>`).
2.  Ensure your virtual environment is activated.
3.  Run: `streamlit run mma_academy_app/app.py`
4.  Access the "Coach Certification" page from the sidebar. You should now see all three levels, with Level 1 available to start.

## Important Notes for Full Functionality

*   **Quiz Setup in DB:** The `8_Coach_Certification.py` page now relies on quizzes being defined in the `Quizzes` table and questions being linked via the `QuizQuestions` table. The example setup in `db_utils.py` (when uncommented and run) provides a starting point for this. You can expand this by adding more quizzes or modifying existing ones directly in the database or through a future admin panel enhancement.
*   **Admin Panel:** The current Admin Panel does not yet have UI for managing certification questions or quiz definitions. This remains a future enhancement.

This version provides a significantly more complete Coach Certification module, ready for users to progress through multiple levels of assessment. Enjoy the enhanced Coach Olympus experience!
