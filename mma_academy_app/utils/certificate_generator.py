"""
Certificate Generation utility for the MMA Coach Certification Module.
Uses WeasyPrint and Jinja2 to create PDF certificates from HTML templates.
"""
import streamlit as st # For st.secrets and potential pathing in Streamlit context
from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import uuid
import datetime

# Assuming this script is in mma_academy_app/utils/
# PROJECT_ROOT_DIR should point to mma_academy_app/
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT_DIR, "templates", "certificates")
ASSETS_DIR = os.path.join(PROJECT_ROOT_DIR, "assets")
USER_CERTIFICATES_DIR = os.path.join(PROJECT_ROOT_DIR, "user_certificates")

# Ensure the output directory exists
os.makedirs(USER_CERTIFICATES_DIR, exist_ok=True)

# Setup Jinja2 environment
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"])
)

def generate_certificate_pdf(user_name: str, level: int, issue_date_str: str, certificate_id: str = None, academy_name: str = None, program_title: str = None, level_specific_title: str = None) -> str | None:
    """
    Generates a PDF certificate for a given user and level.

    Args:
        user_name (str): The full name of the recipient.
        level (int): The certification level (1, 2, or 3).
        issue_date_str (str): The date of issuance in DD/MM/YYYY format.
        certificate_id (str, optional): A pre-defined certificate ID. If None, a new one is generated.
        academy_name (str, optional): The name of the academy. Defaults if None.
        program_title (str, optional): The main title of the program/certificate. Defaults if None.
        level_specific_title (str, optional): The specific title for the level. Defaults if None.

    Returns:
        str | None: The absolute path to the generated PDF file, or None if generation failed.
    """
    if not certificate_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        certificate_id = f"MMA-L{level}-{user_name.split(" ")[-1].upper()[:3]}-{timestamp}-{uuid.uuid4().hex[:4].upper()}"

    template_name = f"level{level}_certificate_template.html"
    try:
        template = jinja_env.get_template(template_name)
    except Exception as e:
        print(f"Error loading template {template_name}: {e}") # Replace with logging
        st.error(f"Certificate template for Level {level} not found or invalid.")
        return None

    # Default titles if not provided
    if academy_name is None:
        academy_name = "Indonesia MMA Youth Excellence Academy & National Coach Training Program"
    
    # Define default titles based on level
    default_titles = {
        1: {
            "program_title": "Certificate of Completion",
            "level_specific_title": "Coach Olympus MMA Fundamentals – Level 1"
        },
        2: {
            "program_title": "Certified Intermediate MMA Coach",
            "level_specific_title": "Level 2 Achievement"
        },
        3: {
            "program_title": "Official Advanced MMA Coach – Certified",
            "level_specific_title": "Level 3 Master Achievement"
        }
    }

    if level not in default_titles:
        st.error(f"Invalid certificate level: {level}. Cannot determine default titles.")
        return None

    current_level_defaults = default_titles[level]
    final_program_title = program_title if program_title is not None else current_level_defaults["program_title"]
    final_level_specific_title = level_specific_title if level_specific_title is not None else current_level_defaults["level_specific_title"]

    logo_filename = "logo.png" # Assuming logo.png is in ASSETS_DIR
    logo_path_for_template = os.path.join(ASSETS_DIR, logo_filename)
    
    if not os.path.exists(logo_path_for_template):
        st.error(f"Logo file not found at {logo_path_for_template}. Certificate generation might fail or look incorrect.")
        logo_path_for_weasyprint = "" # Proceed without logo if not found, or handle as critical error
    else:
        logo_path_for_weasyprint = f"file://{os.path.abspath(logo_path_for_template)}"

    context = {
        "user_name": user_name,
        "academy_name": academy_name,
        "program_title": final_program_title,
        "level_specific_title": final_level_specific_title,
        "issue_date": issue_date_str,
        "certificate_id": certificate_id,
        "logo_path": logo_path_for_weasyprint,
        # TODO: Add paths for badge_path_l2, badge_path_l3, qr_code_path if those assets are created
        # "badge_path_l2": f"file://{os.path.abspath(os.path.join(ASSETS_DIR, 'level2_badge.png'))}", 
        # "badge_path_l3": f"file://{os.path.abspath(os.path.join(ASSETS_DIR, 'level3_badge.png'))}",
        # "qr_code_path": f"file://{os.path.abspath(os.path.join(USER_CERTIFICATES_DIR, 'qr_level3_user.png'))}",
    }

    try:
        html_out = template.render(context)
    except Exception as e:
        print(f"Error rendering HTML template {template_name}: {e}") # Replace with logging
        st.error(f"Could not render certificate details for Level {level}.")
        return None

    safe_user_name = "".join(c if c.isalnum() else "_" for c in user_name)
    pdf_filename = f"Certificate_L{level}_{safe_user_name}_{certificate_id.replace("-", "_").replace(":", "")}.pdf"
    output_pdf_path = os.path.join(USER_CERTIFICATES_DIR, pdf_filename)

    try:
        css_file_path = os.path.join(TEMPLATES_DIR, "certificate_styles.css")
        if not os.path.exists(css_file_path):
            st.warning(f"CSS file not found at {css_file_path}. Certificate might not be styled correctly.")
            stylesheets = None
        else:
            stylesheets = [CSS(css_file_path)]
        
        HTML(string=html_out, base_url=f"file://{TEMPLATES_DIR}/").write_pdf(output_pdf_path, stylesheets=stylesheets)
        print(f"Certificate generated: {output_pdf_path}") # Replace with logging
        return os.path.abspath(output_pdf_path)
    except Exception as e:
        print(f"Error generating PDF for {user_name}, Level {level}: {e}") # Replace with logging
        st.error(f"Failed to generate PDF certificate: {e}")
        return None

if __name__ == "__main__":
    print("Certificate Generator Module - Extended for Levels 1, 2, 3")
    print(f"Project Root: {PROJECT_ROOT_DIR}")
    print(f"Templates Dir: {TEMPLATES_DIR}")
    print(f"Assets Dir: {ASSETS_DIR}")
    print(f"User Certificates Output Dir: {USER_CERTIFICATES_DIR}")

    # Ensure dummy logo exists for testing
    dummy_logo_path = os.path.join(ASSETS_DIR, "logo.png")
    if not os.path.exists(dummy_logo_path):
        print(f"Creating dummy logo at {dummy_logo_path} for testing purposes.")
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (100, 50), color = "maroon")
            d = ImageDraw.Draw(img)
            d.text((10,10), "LOGO", fill=(255,255,0))
            img.save(dummy_logo_path)
        except ImportError:
            print("Pillow not installed, cannot create dummy logo. Please ensure logo.png exists.")
        except Exception as e_logo:
            print(f"Error creating dummy logo: {e_logo}")

    test_user = "Test User Full"
    test_date = datetime.date.today().strftime("%d/%m/%Y")

    for test_level in [1, 2, 3]:
        print(f"\n--- Generating Test Certificate for Level {test_level} --- ")
        generated_pdf_path = generate_certificate_pdf(user_name=test_user, level=test_level, issue_date_str=test_date)
        if generated_pdf_path:
            print(f"Successfully generated Level {test_level} test certificate: {generated_pdf_path}")
        else:
            print(f"Failed to generate Level {test_level} test certificate.")
            print("Check for errors above. Ensure templates (level{test_level}_certificate_template.html), CSS, and assets (logo.png) are correctly placed.")
    print("\nPlease check the user_certificates directory for generated PDFs.")

