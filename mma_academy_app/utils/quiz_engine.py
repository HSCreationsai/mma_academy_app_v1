"""
Quiz Engine for the MMA Coach Certification Module.
Handles loading questions, managing quiz state, and scoring.
"""
import streamlit as st
import json
import os
import random
import datetime

# Assuming the script is in mma_academy_app/utils/
# PROJECT_ROOT_DIR should point to mma_academy_app/
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "questions")

class QuizEngine:
    def __init__(self, user_id):
        self.user_id = user_id
        self.current_quiz_id = None
        self.current_quiz_questions = []
        self.current_question_index = 0
        self.user_answers = {}
        self.quiz_start_time = None
        self.quiz_definition = None # Stores the Quiz object from DB

    def load_quiz(self, quiz_id: int, quiz_definition: dict, questions: list):
        """Loads a specific quiz for the user."""
        self.current_quiz_id = quiz_id
        self.quiz_definition = quiz_definition
        self.current_quiz_questions = questions # Expects a list of question dicts
        self.current_question_index = 0
        self.user_answers = {q["id"]: None for q in self.current_quiz_questions}
        self.quiz_start_time = datetime.datetime.now()
        # Store in session state for persistence across reruns
        st.session_state[f"quiz_engine_{self.user_id}_{quiz_id}"] = self.__dict__

    @classmethod
    def get_quiz_session(cls, user_id: int, quiz_id: int):
        """Retrieves an existing quiz session or creates a new one."""
        session_key = f"quiz_engine_{user_id}_{quiz_id}"
        if session_key in st.session_state:
            engine = cls(user_id)
            engine.__dict__.update(st.session_state[session_key])
            return engine
        return None # Or raise an error, or create new - depends on desired flow

    def get_current_question(self):
        if 0 <= self.current_question_index < len(self.current_quiz_questions):
            return self.current_quiz_questions[self.current_question_index]
        return None

    def record_answer(self, question_id: str, answer_id):
        if question_id in self.user_answers:
            self.user_answers[question_id] = answer_id
            st.session_state[f"quiz_engine_{self.user_id}_{self.current_quiz_id}"] = self.__dict__

    def next_question(self):
        if self.current_question_index < len(self.current_quiz_questions) - 1:
            self.current_question_index += 1
            st.session_state[f"quiz_engine_{self.user_id}_{self.current_quiz_id}"] = self.__dict__
            return True
        return False

    def previous_question(self):
        if self.current_question_index > 0:
            self.current_question_index -= 1
            st.session_state[f"quiz_engine_{self.user_id}_{self.current_quiz_id}"] = self.__dict__
            return True
        return False

    def calculate_score(self):
        score = 0
        total_marks_possible = 0
        correct_answers = 0
        
        for question in self.current_quiz_questions:
            q_id = question["id"]
            total_marks_possible += question.get("marks", 1)
            user_ans = self.user_answers.get(q_id)
            correct_ans = question["correct_answer_id"]
            
            if isinstance(correct_ans, list): # For multi-select (not in current simple MCQ)
                if isinstance(user_ans, list) and sorted(user_ans) == sorted(correct_ans):
                    score += question.get("marks", 1)
                    correct_answers +=1
            elif user_ans == correct_ans:
                score += question.get("marks", 1)
                correct_answers +=1
        
        percentage = (score / total_marks_possible) * 100 if total_marks_possible > 0 else 0
        return {
            "score_achieved": score,
            "total_marks_possible": total_marks_possible,
            "questions_attempted": len(self.current_quiz_questions),
            "questions_correct": correct_answers,
            "percentage": round(percentage, 2)
        }

    def is_quiz_passed(self, achieved_percentage: float):
        if self.quiz_definition:
            return achieved_percentage >= self.quiz_definition.get("passing_score_percentage", 75.0)
        return False # Default if quiz definition not loaded

    def get_quiz_duration_seconds(self):
        if self.quiz_start_time:
            return (datetime.datetime.now() - self.quiz_start_time).total_seconds()
        return 0

# --- Helper functions for loading questions from JSON files --- 

def load_questions_from_file(file_path: str) -> list:
    """Loads a list of question dictionaries from a single JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
            if isinstance(questions_data, list):
                return questions_data
            # If the JSON root is a dict with a key like "questions"
            elif isinstance(questions_data, dict) and "questions" in questions_data:
                return questions_data["questions"]
            else:
                st.error(f"Invalid question format in {file_path}. Expected a list of questions or a dict with a 'questions' key.")
                return []
    except FileNotFoundError:
        st.error(f"Question file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading questions from {file_path}: {e}")
        return []

def get_questions_for_quiz(quiz_definition: dict, all_questions_db: list) -> list:
    """ 
    Selects questions for a given quiz definition.
    - If quiz_definition has specific question_ids, fetch those.
    - Else, if it has source_topic_file and num_questions, sample from that topic.
    `all_questions_db` is a list of all question dicts loaded from the database (Questions table).
    """
    # This function will be more elaborate once DB interaction for QuizQuestions link table is set up.
    # For now, let's assume quiz_definition might point to a topic file and num_questions.
    
    selected_questions = []
    source_topic_file = quiz_definition.get("source_topic_file") # e.g., "manual_basics.json"
    num_questions_to_select = quiz_definition.get("num_questions")
    level = quiz_definition.get("level")

    if source_topic_file and num_questions_to_select and level:
        # Construct path to the JSON file
        file_path = os.path.join(QUESTIONS_BASE_DIR, f"level{level}", source_topic_file)
        topic_questions = load_questions_from_file(file_path)
        
        if len(topic_questions) >= num_questions_to_select:
            selected_questions = random.sample(topic_questions, num_questions_to_select)
        else:
            st.warning(f"Not enough questions in {source_topic_file} ({len(topic_questions)}) to select {num_questions_to_select}. Using all available.")
            selected_questions = topic_questions
    else:
        # This part would fetch from `QuizQuestions` link table if using that approach
        st.error("Quiz definition is missing details to select questions (source_topic_file, num_questions, level).")

    return selected_questions

# Example Usage (Conceptual - to be called from Streamlit page)
# def start_a_quiz(user_id, quiz_id_from_db, quiz_definition_from_db, all_questions_from_db):
#     engine = QuizEngine(user_id)
#     questions_for_this_quiz = get_questions_for_quiz(quiz_definition_from_db, all_questions_from_db)
#     if questions_for_this_quiz:
#         engine.load_quiz(quiz_id_from_db, quiz_definition_from_db, questions_for_this_quiz)
#         st.session_state.active_quiz_engine = engine
#         return engine
#     return None

if __name__ == "__main__":
    # Basic test for loading questions from a dummy file
    print(f"Project Root: {PROJECT_ROOT_DIR}")
    print(f"Questions Base: {QUESTIONS_BASE_DIR}")
    
    # Create dummy question file for testing
    dummy_level_dir = os.path.join(QUESTIONS_BASE_DIR, "level1")
    os.makedirs(dummy_level_dir, exist_ok=True)
    dummy_q_file = os.path.join(dummy_level_dir, "dummy_test.json")
    dummy_questions_content = [
        {
            "id": "L1DT001", "level": 1, "topic": "Dummy Test", "question_type": "mcq",
            "text": "What is 1+1?", "options": [{"id": "A", "text": "1"}, {"id": "B", "text": "2"}],
            "correct_answer_id": "B", "marks": 1
        },
        {
            "id": "L1DT002", "level": 1, "topic": "Dummy Test", "question_type": "true_false",
            "text": "Is the sky blue?", "correct_answer_id": "True", "marks": 1 # Assuming "True"/"False" as string for consistency
        }
    ]
    with open(dummy_q_file, "w") as f:
        json.dump(dummy_questions_content, f, indent=2)

    loaded_qs = load_questions_from_file(dummy_q_file)
    print(f"Loaded {len(loaded_qs)} questions from {dummy_q_file}:")
    for q in loaded_qs:
        print(q)

    # Clean up dummy file
    # os.remove(dummy_q_file)
    print("\nQuiz Engine module structure created. Further testing requires Streamlit context or DB integration.")

