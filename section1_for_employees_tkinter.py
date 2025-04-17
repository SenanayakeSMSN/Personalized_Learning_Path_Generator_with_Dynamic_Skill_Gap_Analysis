# Cell 1: Install Dependencies
#pip install pdfplumber pymongo google-generativeai dnspython pytesseract Pillow

# Cell 2: Import Libraries and Setup
import os
import json
import logging
import re
from pymongo import MongoClient
from PIL import Image
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import logging.handlers

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcode credentials
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo"  # Your provided Gemini API key
MONGODB_URI = "mongodb+srv://shashi:VSXV9WDNmRvYnA7p@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority"

# Import dependencies
try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is not installed. Run: %pip install pdfplumber")

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("google-generativeai is not installed. Run: %pip install google-generativeai")

try:
    import pytesseract
    # Explicitly set Tesseract path (adjust if installed elsewhere)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    raise ImportError("pytesseract or PIL is not installed. Run: %pip install pytesseract Pillow")

# Cell 3: MongoDB Connection
def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
        collection.create_index("Job_Role")
        job_roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1})]
        logging.info("Connected to MongoDB Atlas")
        return collection, job_roles
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

# Cell 4: Enhanced CV Text Extraction
def extract_cv_text(cv_path):
    try:
        with pdfplumber.open(cv_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"  # Double newline to separate pages
                    logging.info(f"Extracted text from page {page.page_number} using pdfplumber")
                else:
                    logging.warning(f"No text extracted from page {page.page_number}. Using OCR.")
                    try:
                        img = page.to_image().original
                        ocr_text = pytesseract.image_to_string(img)
                        text += ocr_text + "\n\n"
                        logging.info(f"Extracted OCR text from page {page.page_number}")
                    except Exception as ocr_e:
                        logging.error(f"OCR failed for page {page.page_number}: {str(ocr_e)}")
                        text += "\n\n"
            if not text.strip():
                raise ValueError("No text extracted from CV. Check PDF format or Tesseract installation.")
            logging.info(f"Extracted text from CV: {text[:100]}...")
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error("PDF is password-protected. Provide the password or use an unprotected PDF.")
        raise ValueError("PDF is password-protected")
    except Exception as e:
        logging.error(f"Error reading CV: {str(e)}")
        raise ValueError(f"Error reading CV: {str(e)}")

# Cell 5: Skills Section Extraction
def extract_skills_section(cv_text):
    try:
        # Search for "Skills" section anywhere in the CV
        match = re.search(
            r"(skills|technical skills|key skills|core competencies):?\s*(.*?)(?=\n\s*(experience|education|projects|contact|certifications|references|$|\n\n))",
            cv_text, re.IGNORECASE | re.DOTALL
        )
        if match:
            skills_text = match.group(2).strip()
            # Include all non-empty lines
            skills_lines = [line.strip() for line in skills_text.split('\n') if line.strip() and not re.match(r'^\s*$', line)]
            cleaned_skills = ' '.join(skills_lines).strip()
            logging.info(f"Extracted skills section: {cleaned_skills}")
            return cleaned_skills if cleaned_skills else cv_text
        logging.warning("No explicit skills section found. Using full CV text for Gemini.")
        return cv_text  # Fall back to full CV text
    except Exception as e:
        logging.error(f"Error extracting skills section: {str(e)}")
        return cv_text

# Cell 6: Gemini Skill Extraction
def extract_skills_with_gemini(cv_text):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert in CV analysis. Extract all skills from the following CV text, including: "
            "1. Explicit skills listed in a 'Skills', 'Technical Skills', or similar section (e.g., 'Python, TensorFlow'). "
            "2. Implicit skills inferred from 'Projects', 'Experience', or similar sections (e.g., 'Built a fraud detection model using scikit-learn' implies 'scikit-learn'). "
            "Return a plain JSON array of clean, distinct skills (e.g., capitalize 'PyTorch', 'AWS SageMaker'). "
            "Combine similar skills (e.g., 'ML' and 'Machine Learning' as 'Machine Learning'). "
            "Return [] if no skills are found or input is empty. "
            "Output must be a valid JSON array without ```json, backticks, or any other formatting. "
            "Example: Input: 'Skills: Python, ML\nProjects: Built a model using TensorFlow' "
            "Output: [\"Python\", \"Machine Learning\", \"TensorFlow\"]"
            "\n\nCV Text:\n" + cv_text
        )
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw response: {raw_response}")
        cleaned_response = raw_response
        if cleaned_response.startswith("```json\n"):
            cleaned_response = cleaned_response[8:].strip()
        elif cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.endswith("\n```"):
            cleaned_response = cleaned_response[:-4].strip()
        elif cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
        logging.info(f"Cleaned Gemini response: {cleaned_response}")
        try:
            skills = json.loads(cleaned_response)
            if not isinstance(skills, list):
                logging.error("Gemini response is not a JSON array")
                return []
            logging.info(f"Extracted skills: {skills}")
            return skills
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse cleaned Gemini response as JSON: {cleaned_response}, Error: {str(e)}")
            return []
    except Exception as e:
        logging.error(f"Error in Gemini skill extraction: {str(e)}")
        return []

# Cell 7: MongoDB Query
def get_required_skills(job_collection, job_role):
    try:
        job_doc = job_collection.find_one({"Job_Role": job_role})
        if not job_doc:
            raise ValueError(f"Job role '{job_role}' not found in database")
        required_skills = [skill.strip() for skill in job_doc["Required_Skills"].split(",")]
        logging.info(f"Required skills for {job_role}: {required_skills}")
        return required_skills
    except Exception as e:
        logging.error(f"Error retrieving required skills: {str(e)}")
        raise ValueError(f"Error retrieving required skills: {str(e)}")

# Cell 8: Skill Gap Analysis with Gemini Semantic Similarity
def skill_gap_analysis(cv_skills, required_skills):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert in skill gap analysis. Given two lists of skills: "
            "1. CV skills (from a candidate's CV). "
            "2. Required skills (for a job role). "
            "Identify which required skills are missing from the CV skills, considering semantic similarity and synonyms. "
            "Treat skills as equivalent if they have the same or similar meaning (e.g., 'ML' ≈ 'Machine Learning', "
            "'SQL' ≈ 'Database Management', 'Statistical Analysis' ≈ 'Statistics', 'Deep Learning' ≈ 'Neural Networks'). "
            "Return a plain JSON array of the missing required skills, preserving their original names from the required skills list. "
            "Output must be a valid JSON array without ```json or backticks. "
            "Example: "
            "CV skills: ['Python', 'SQL', 'Deep Learning'] "
            "Required skills: ['Database Management', 'Neural Networks', 'Data Pipelines'] "
            "Output: ['Data Pipelines'] "
            "\n\nCV Skills:\n" + json.dumps(cv_skills) +
            "\nRequired Skills:\n" + json.dumps(required_skills)
        )
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw response for skill gap: {raw_response}")
        cleaned_response = raw_response
        if cleaned_response.startswith("```json\n"):
            cleaned_response = cleaned_response[8:].strip()
        elif cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.endswith("\n```"):
            cleaned_response = cleaned_response[:-4].strip()
        elif cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
        logging.info(f"Cleaned Gemini response for skill gap: {cleaned_response}")
        try:
            missing_skills = json.loads(cleaned_response)
            if not isinstance(missing_skills, list):
                logging.error("Gemini response is not a JSON array")
                return []
            logging.info(f"Missing skills: {missing_skills}")
            return missing_skills
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini response: {cleaned_response}, Error: {str(e)}")
            return []
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        raise ValueError(f"Error in skill gap analysis: {str(e)}")

# Cell 9: Main Analysis Function
def analyze_cv_skill_gap(cv_path, dream_job_role):
    try:
        job_collection, _ = connect_to_mongodb()
        cv_text = extract_cv_text(cv_path)
        skills_text = extract_skills_section(cv_text)
        cv_skills = extract_skills_with_gemini(skills_text)
        if not cv_skills:
            logging.error("Failed to extract skills from CV")
            return {"error": "Failed to extract skills from CV. Check CV formatting, content, or Gemini API response."}
        required_skills = get_required_skills(job_collection, dream_job_role)
        missing_skills = skill_gap_analysis(cv_skills, required_skills)
        result = {
            "job_role": dream_job_role,
            "cv_skills": cv_skills,
            "required_skills": required_skills,
            "missing_skills": missing_skills
        }
        logging.info(f"Skill gap analysis result: {result}")
        return result
    except Exception as e:
        logging.error(f"Skill gap analysis error: {str(e)}")
        return {"error": str(e)}

# Cell 10: Tkinter GUI
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.config(state='disabled')
        self.text_widget.yview(tk.END)

def create_gui():
    root = tk.Tk()
    root.title("Skill Gap Analysis")
    root.geometry("600x500")

    # CV File Selection
    tk.Label(root, text="Select PDF CV:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    cv_entry = tk.Entry(root, width=50)
    cv_entry.grid(row=0, column=1, padx=5, pady=5)

    def browse_cv():
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            cv_entry.delete(0, tk.END)
            cv_entry.insert(0, file_path)

    tk.Button(root, text="Browse", command=browse_cv).grid(row=0, column=2, padx=5, pady=5)

    # Job Role Selection
    tk.Label(root, text="Select Job Role:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    job_role_var = tk.StringVar()
    try:
        _, job_roles = connect_to_mongodb()
        if not job_roles:
            job_roles = ["AI Governance Expert"]
    except Exception as e:
        job_roles = ["AI Governance Expert"]
        logging.error(f"Failed to fetch job roles: {str(e)}")
    job_role_var.set(job_roles[0])
    job_role_dropdown = ttk.OptionMenu(root, job_role_var, job_roles[0], *job_roles)
    job_role_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    # Results Area
    result_text = scrolledtext.ScrolledText(root, width=70, height=20, state='disabled')
    result_text.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    # Redirect logging to text area
    text_handler = TextHandler(result_text)
    text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().handlers = [text_handler]

    # Run Analysis
    def run_analysis():
        cv_path = cv_entry.get()
        job_role = job_role_var.get()
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.config(state='disabled')
        if not cv_path or not job_role:
            logging.error("Please select a PDF CV file and job role.")
            return
        try:
            result = analyze_cv_skill_gap(cv_path, job_role)
            result_text.config(state='normal')
            result_text.insert(tk.END, f"Analysis Result:\n{json.dumps(result, indent=2)}\n")
            result_text.config(state='disabled')
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")

    tk.Button(root, text="Run Analysis", command=run_analysis).grid(row=2, column=1, padx=5, pady=5)

    root.mainloop()

# Cell 11: Run the GUI
create_gui()