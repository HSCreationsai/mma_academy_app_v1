"""
Utility to convert the PDF manual to a JSON file for the RAG agent and Manual Reader.
Extracts text page by page or by chapter (page by page is simpler initially).
"""
import PyPDF2
import json
import os
import streamlit as st # For st.secrets if needed, though not directly here

# Define paths relative to the mma_academy_app directory
# Assuming this script is in mma_academy_app/utils/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_ASSET_PATH = os.path.join(BASE_DIR, "assets", "manual_draft.pdf") # Source PDF
JSON_DATA_PATH = os.path.join(BASE_DIR, "data", "manual.json")    # Output JSON

def convert_pdf_to_json(pdf_path=PDF_ASSET_PATH, json_path=JSON_DATA_PATH):
    """Reads a PDF file and saves its content as a JSON object, page by page."""
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        # st.error(f"Error: PDF file not found at {pdf_path}") # If run from Streamlit context
        return False

    manual_data = {}
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
            print(f"Found {num_pages} pages in the PDF.")

            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text: # Ensure there is text to add
                    manual_data[f"Page {i + 1}"] = text.strip()
                else:
                    manual_data[f"Page {i + 1}"] = "[No text extracted from this page. It might be an image or scanned document.]"
                if (i+1) % 20 == 0:
                    print(f"Processed {i+1}/{num_pages} pages...")
            
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(manual_data, json_file, ensure_ascii=False, indent=4)
        
        print(f"Successfully converted PDF to JSON: {json_path}")
        # st.success(f"Successfully converted PDF to JSON: {json_path}") # If run from Streamlit context
        return True

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        # st.error(f"PDF file not found: {pdf_path}")
        return False
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file: {e}. The PDF might be encrypted or corrupted.")
        # st.error(f"Error reading PDF file: {e}. The PDF might be encrypted or corrupted.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during PDF to JSON conversion: {e}")
        # st.error(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Starting PDF to JSON conversion process...")
    # Ensure the script is run from the project root or adjust paths accordingly if run directly.
    # For this script, paths are relative to its location in utils/.
    
    # Check if assets/manual_draft.pdf exists
    if not os.path.exists(PDF_ASSET_PATH):
        print(f"CRITICAL ERROR: The manual PDF expected at 	{PDF_ASSET_PATH} was not found.")
        print("Please ensure the main training manual PDF is placed at that location and named 'manual_draft.pdf'.")
        print("You might need to copy 'complete_manual complete 07.052025.pdf' to 'assets/manual_draft.pdf'.")
    else:
        success = convert_pdf_to_json()
        if success:
            print("PDF to JSON conversion completed successfully.")
        else:
            print("PDF to JSON conversion failed. Check errors above.")

