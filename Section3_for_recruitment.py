# Cell 1: Install Dependencies
#pip install pdfplumber pymongo google-generativeai dnspython pytesseract Pillow

# Cell 2: Import Libraries and Setup
import os
import json
import logging
import re
from datetime import datetime
from pymongo import MongoClient
from PIL import Image

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
                    text += page_text + "\n\n"
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
        match = re.search(
            r"(skills|technical skills|key skills|core competencies):?\s*(.*?)(?=\n\s*(experience|education|projects|contact|certifications|references|$|\n\n))",
            cv_text, re.IGNORECASE | re.DOTALL
        )
        if match:
            skills_text = match.group(2).strip()
            skills_lines = [line.strip() for line in skills_text.split('\n') if line.strip() and not re.match(r'^\s*$', line)]
            cleaned_skills = ' '.join(skills_lines).strip()
            logging.info(f"Extracted skills section: {cleaned_skills}")
            return cleaned_skills if cleaned_skills else cv_text
        logging.warning("No explicit skills section found. Using full CV text for Gemini.")
        return cv_text
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
        return []

# Cell 9: Functions for Experience, Skill Weighting, and Ranking
def extract_experience(cv_text):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert in CV analysis. Extract the total years of professional experience from the following CV text by analyzing the 'Experience' or 'Work Experience' section. "
            "Identify all job entries and their date ranges (e.g., 'March 2016 - June 2018'). "
            "For each job, calculate the duration in months: (end_year - start_year)*12 + (end_month - start_month). "
            "If the end date is 'Present', use April 2025 as the end date. "
            "If a date is ambiguous (e.g., only year provided), assume January for start month and December for end month. "
            "If a job has no clear date range, estimate 1 year if it seems like a short-term role (e.g., intern), else skip it. "
            "Sum the durations of all jobs, convert to years (divide by 12, round to nearest integer), and return a single integer. "
            "If no experience or dates are found, return 0. "
            "Output only the integer, nothing else. "
            "Example: "
            "Input: 'Data Scientist, March 2016 - June 2018\nAnalyst, April 2015 - March 2016' "
            "Calculation: (2018-2016)*12 + (6-3) = 27 months; (2016-2015)*12 + (3-4) = 11 months; Total = 38/12 ≈ 3 years "
            "Output: 3 "
            "\n\nCV Text:\n" + cv_text
        )
        response = model.generate_content(prompt)
        years = int(response.text.strip())
        logging.info(f"Gemini extracted experience: {years} years")
        return years
    except Exception as e:
        logging.error(f"Error extracting experience: {str(e)}")
        return 0

def assign_experience_weight(years):
    if years > 15:
        return 3  # High
    elif 8 <= years <= 15:
        return 2  # Medium
    elif 1 <= years <= 7:
        return 1  # Low
    return 0  # None or <1 year

def assign_skill_weights(job_role, required_skills):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        num_skills = len(required_skills)
        prompt = (
            f"You are an expert in recruitment for the job role '{job_role}'. "
            f"Given the following required skills: {json.dumps(required_skills)}, "
            f"assign a weight to each skill based on its relevance to the job role. "
            f"Use integers from 1 (least relevant) to {num_skills} (most relevant), ensuring each skill gets a unique weight. "
            "Consider technical skills (e.g., 'Python' for Data Scientist) as more relevant than soft or unrelated skills (e.g., 'Hiking'). "
            "Return a JSON object mapping each skill to its weight. "
            "Output must be a valid JSON object without ```json or backticks. "
            "Example: "
            "Job role: Data Scientist "
            "Required skills: ['Python', 'Report Writing', 'Hiking'] "
            "Output: {\"Python\": 3, \"Report Writing\": 2, \"Hiking\": 1}"
            "\n\nRequired Skills:\n" + json.dumps(required_skills)
        )
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw response for skill weights: {raw_response}")
        cleaned_response = raw_response
        if cleaned_response.startswith("```json\n"):
            cleaned_response = cleaned_response[8:].strip()
        elif cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.endswith("\n```"):
            cleaned_response = cleaned_response[:-4].strip()
        elif cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
        logging.info(f"Cleaned Gemini response for skill weights: {cleaned_response}")
        try:
            weights = json.loads(cleaned_response)
            if not isinstance(weights, dict):
                logging.error("Gemini response is not a JSON object")
                return {skill: 1 for skill in required_skills}  # Fallback: equal weights
            logging.info(f"Skill weights: {weights}")
            return weights
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini response: {cleaned_response}, Error: {str(e)}")
            return {skill: 1 for skill in required_skills}
    except Exception as e:
        logging.error(f"Error assigning skill weights: {str(e)}")
        return {skill: 1 for skill in required_skills}

def rank_cvs(cv_folder, job_role):
    try:
        job_collection, _ = connect_to_mongodb()
        required_skills = get_required_skills(job_collection, job_role)
        skill_weights = assign_skill_weights(job_role, required_skills)
        cv_results = []

        for cv_file in os.listdir(cv_folder):
            if cv_file.lower().endswith('.pdf'):
                cv_path = os.path.join(cv_folder, cv_file)
                try:
                    cv_text = extract_cv_text(cv_path)
                    skills_text = extract_skills_section(cv_text)
                    cv_skills = extract_skills_with_gemini(skills_text)
                    if not cv_skills:
                        logging.warning(f"No skills extracted from {cv_file}")
                        continue

                    years = extract_experience(cv_text)
                    exp_weight = assign_experience_weight(years)
                    missing_skills = skill_gap_analysis(cv_skills, required_skills)
                    skill_score = sum(skill_weights.get(skill, 0) for skill in cv_skills if skill in required_skills)
                    total_score = exp_weight + skill_score

                    cv_results.append({
                        "cv_file": cv_file,
                        "years_experience": years,
                        "experience_weight": exp_weight,
                        "cv_skills": cv_skills,
                        "required_skills": required_skills,
                        "missing_skills": missing_skills,
                        "skill_score": skill_score,
                        "total_score": total_score
                    })
                    logging.info(f"Processed {cv_file}: Score = {total_score}")
                except Exception as e:
                    logging.error(f"Error processing {cv_file}: {str(e)}")
                    continue

        ranked_cvs = sorted(cv_results, key=lambda x: x["total_score"], reverse=True)
        logging.info(f"Ranked {len(ranked_cvs)} CVs")
        return ranked_cvs
    except Exception as e:
        logging.error(f"Error in CV ranking: {str(e)}")
        return []

# Cell 10: Tkinter GUI
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

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
    root.title("CV Ranking System")
    root.geometry("800x600")

    # CV Folder Selection
    tk.Label(root, text="Select CV Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    folder_entry = tk.Entry(root, width=60)
    folder_entry.grid(row=0, column=1, padx=5, pady=5)

    def browse_folder():
        folder_path = filedialog.askdirectory()
        if folder_path:
            folder_entry.delete(0, tk.END)
            folder_entry.insert(0, folder_path)

    tk.Button(root, text="Browse", command=browse_folder).grid(row=0, column=2, padx=5, pady=5)

    # Job Role Selection
    tk.Label(root, text="Select Job Role:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    job_role_var = tk.StringVar()
    try:
        _, job_roles = connect_to_mongodb()
        if not job_roles:
            job_roles = ["Data Scientist"]
    except Exception as e:
        job_roles = ["Data Scientist"]
        logging.error(f"Failed to fetch job roles: {str(e)}")
    job_role_var.set(job_roles[0])
    job_role_dropdown = ttk.OptionMenu(root, job_role_var, job_roles[0], *job_roles)
    job_role_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    # Results Area
    result_text = scrolledtext.ScrolledText(root, width=90, height=30, state='disabled')
    result_text.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    # Redirect logging to text area
    text_handler = TextHandler(result_text)
    text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().handlers = [text_handler]

    # Run Analysis
    def run_analysis():
        cv_folder = folder_entry.get()
        job_role = job_role_var.get()
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.config(state='disabled')
        if not cv_folder or not job_role:
            logging.error("Please select a CV folder and job role.")
            return
        try:
            ranked_cvs = rank_cvs(cv_folder, job_role)
            result_text.config(state='normal')
            result_text.insert(tk.END, "Ranked CVs:\n")
            for i, cv in enumerate(ranked_cvs, 1):
                result_text.insert(tk.END, f"Rank {i}: {cv['cv_file']}\n")
                result_text.insert(tk.END, f"  Years of Experience: {cv['years_experience']} (Weight: {cv['experience_weight']})\n")
                result_text.insert(tk.END, f"  CV Skills: {cv['cv_skills']}\n")
                result_text.insert(tk.END, f"  Missing Skills: {cv['missing_skills']}\n")
                result_text.insert(tk.END, f"  Skill Score: {cv['skill_score']}\n")
                result_text.insert(tk.END, f"  Total Score: {cv['total_score']}\n\n")
            result_text.config(state='disabled')
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")

    tk.Button(root, text="Run Analysis", command=run_analysis).grid(row=2, column=1, padx=5, pady=5)

    root.mainloop()

create_gui()