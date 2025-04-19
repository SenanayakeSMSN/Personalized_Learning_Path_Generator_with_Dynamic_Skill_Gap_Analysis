
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
MONGODB_URI = "mongodb+srv://akilapremarathna0:123@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority"

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
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    raise ImportError("langchain-google-genai is not installed. Run: %pip install langchain-google-genai")

try:
    import pytesseract
    # Explicitly set Tesseract path (adjust if installed elsewhere)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    raise ImportError("pytesseract or PIL is not installed. Run: %pip install pytesseract Pillow")

from langchain_core.messages import HumanMessage, SystemMessage

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

# Cell 6: OPTIMIZED - Combined function to extract both skills and experience in a single API call
def extract_cv_data_with_gemini(cv_text):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert in CV analysis. Extract the following from the CV text:\n\n"
            "1. SKILLS: Extract all skills including explicit skills (from 'Skills' sections) and implicit skills (inferred from experience/projects).\n"
            "2. EXPERIENCE: Calculate total years of professional experience by analyzing all job entries and their date ranges.\n\n"
            "For SKILLS:\n"
            "- Return distinct skills (e.g., capitalize 'PyTorch', 'AWS SageMaker').\n"
            "- Combine similar skills (e.g., 'ML' and 'Machine Learning' as 'Machine Learning').\n\n"
            "For EXPERIENCE:\n"
            "- Calculate duration for each job: (end_year - start_year)*12 + (end_month - start_month).\n"
            "- If end date is 'Present', use April 2025.\n"
            "- For ambiguous dates (only year), assume January for start and December for end.\n"
            "- Sum all durations, convert to years (divide by 12, round to nearest integer).\n\n"
            "Return a JSON object with exactly two keys:\n"
            "- 'skills': array of skill strings\n"
            "- 'years': integer representing years of experience\n\n"
            "Example output: {\"skills\": [\"Python\", \"Machine Learning\", \"TensorFlow\"], \"years\": 3}\n\n"
            "Your output must be a valid JSON object without backticks or formatting.\n\n"
            "CV Text:\n" + cv_text
        )
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw combined response: {raw_response[:100]}...")
        
        # Clean response
        cleaned_response = raw_response
        for pattern in [r"```json\n", r"```json", r"\n```", r"```"]:
            cleaned_response = re.sub(pattern, "", cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        try:
            cv_data = json.loads(cleaned_response)
            if not isinstance(cv_data, dict) or "skills" not in cv_data or "years" not in cv_data:
                logging.error("Gemini response format invalid for CV data")
                return [], 0
                
            skills = cv_data.get("skills", [])
            years = int(cv_data.get("years", 0))
            
            logging.info(f"Extracted {len(skills)} skills and {years} years experience")
            return skills, years
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini response: {cleaned_response[:100]}, Error: {str(e)}")
            return [], 0
    except Exception as e:
        logging.error(f"Error in combined CV data extraction: {str(e)}")
        return [], 0

# Cell 7: MongoDB Query for Required Skills
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

# Cell 8: Experience Weight Assignment
def assign_experience_weight(years):
    if years > 15:
        return 3  # High
    elif 8 <= years <= 15:
        return 2  # Medium
    elif 1 <= years <= 7:
        return 1  # Low
    return 0  # None or <1 year

# Cell 9: OPTIMIZED - Combined function to analyze skill weights and skill gaps in a single API call
def analyze_job_requirements(job_role, required_skills, cv_skills):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        num_skills = len(required_skills)
        prompt = (
            f"You are an expert in recruitment for the job role '{job_role}'.\n\n"
            f"TASK 1 - SKILL WEIGHTS:\n"
            f"For the required skills: {json.dumps(required_skills)},\n"
            f"assign a weight to each skill based on its relevance to the '{job_role}' role.\n"
            f"Use integers from 1 (least relevant) to {num_skills} (most relevant), ensuring each skill gets a unique weight.\n"
            f"Technical skills specific to the role should receive higher weights.\n\n"
            f"TASK 2 - SKILL GAP ANALYSIS:\n"
            f"Compare the CV skills: {json.dumps(cv_skills)}\n"
            f"with the required skills: {json.dumps(required_skills)}\n"
            f"Identify which required skills are missing from the CV skills, considering semantic similarity.\n"
            f"For example, 'ML' ≈ 'Machine Learning', 'SQL' ≈ 'Database Management'.\n\n"
            f"Return a JSON object with exactly two keys:\n"
            f"- 'weights': object mapping each required skill to its weight (integer)\n"
            f"- 'missing_skills': array of required skills missing from CV\n\n"
            f"Example: {{\"weights\": {{\"Python\": 3, \"Machine Learning\": 2, \"SQL\": 1}}, \"missing_skills\": [\"SQL\"]}}\n\n"
            f"Output must be a valid JSON object without backticks or formatting."
        )
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw combined job analysis response: {raw_response[:100]}...")
        
        # Clean response
        cleaned_response = raw_response
        for pattern in [r"```json\n", r"```json", r"\n```", r"```"]:
            cleaned_response = re.sub(pattern, "", cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        try:
            analysis_data = json.loads(cleaned_response)
            if not isinstance(analysis_data, dict) or "weights" not in analysis_data or "missing_skills" not in analysis_data:
                logging.error("Gemini response format invalid for job analysis")
                return {skill: 1 for skill in required_skills}, []
                
            weights = analysis_data.get("weights", {})
            missing_skills = analysis_data.get("missing_skills", [])
            
            logging.info(f"Analyzed weights for {len(weights)} skills and found {len(missing_skills)} missing skills")
            return weights, missing_skills
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini response: {cleaned_response[:100]}, Error: {str(e)}")
            return {skill: 1 for skill in required_skills}, []
    except Exception as e:
        logging.error(f"Error in combined job analysis: {str(e)}")
        return {skill: 1 for skill in required_skills}, []

# Cell 10: Updated CV Ranking Function using optimized API calls
def rank_cvs(cv_folder, job_role):
    try:
        job_collection, _ = connect_to_mongodb()
        required_skills = get_required_skills(job_collection, job_role)
        cv_results = []
        cv_texts = {}  # Store CV text for chatbot

        for cv_file in os.listdir(cv_folder):
            if cv_file.lower().endswith('.pdf'):
                cv_path = os.path.join(cv_folder, cv_file)
                try:
                    cv_text = extract_cv_text(cv_path)
                    cv_texts[cv_file] = cv_text  # Store CV text
                    skills_text = extract_skills_section(cv_text)
                    
                    # First combined API call - extract skills and experience
                    cv_skills, years = extract_cv_data_with_gemini(skills_text)
                    if not cv_skills:
                        logging.warning(f"No skills extracted from {cv_file}")
                        continue

                    exp_weight = assign_experience_weight(years)
                    
                    # Second combined API call - analyze skill weights and gaps
                    skill_weights, missing_skills = analyze_job_requirements(job_role, required_skills, cv_skills)
                    
                    skill_score = sum(skill_weights.get(skill, 0) for skill in cv_skills if skill in required_skills)
                    # Add a skill match percentage for better analysis
                    matched_skills = [skill for skill in cv_skills if skill in required_skills]
                    skill_match_percent = len(matched_skills) / len(required_skills) * 100 if required_skills else 0
                    total_score = exp_weight + skill_score

                    cv_results.append({
                        "cv_file": cv_file,
                        "years_experience": years,
                        "experience_weight": exp_weight,
                        "cv_skills": cv_skills,
                        "required_skills": required_skills,
                        "missing_skills": missing_skills,
                        "skill_score": skill_score,
                        "skill_match_percent": round(skill_match_percent, 1),
                        "total_score": total_score
                    })
                    logging.info(f"Processed {cv_file}: Score = {total_score}, Match = {round(skill_match_percent, 1)}%")
                except Exception as e:
                    logging.error(f"Error processing {cv_file}: {str(e)}")
                    continue

        ranked_cvs = sorted(cv_results, key=lambda x: x["total_score"], reverse=True)
        logging.info(f"Ranked {len(ranked_cvs)} CVs")
        return ranked_cvs, cv_texts
    except Exception as e:
        logging.error(f"Error in CV ranking: {str(e)}")
        return [], {}

# Cell 11: Tkinter GUI with Chatbot
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
    root.title("CV Ranking System with Chatbot")
    root.geometry("1000x700")

    # Initialize LangChain Gemini model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    except Exception as e:
        logging.error(f"Failed to initialize Gemini LLM: {str(e)}")
        llm = None

    # Define ask_cv_question inside create_gui to access ranked_cvs and job_role_var
    def ask_cv_question(cv_text, question, selected_cv):
        try:
            job_role = job_role_var.get()
            # Create analysis_json
            analysis_json = json.dumps(ranked_cvs, indent=2)
            # Relevance check
            relevance_prompt = [
                SystemMessage(content="You are an assistant that determines if a question is related to a CV or the analysis of CVs for a job role. Questions about skills, experience, education, projects, certifications, ranking, scores, missing skills, etc., are considered related. Respond with only 'yes' or 'no'."),
                HumanMessage(content=f"Question: {question}")
            ]
            relevance_response = llm.invoke(relevance_prompt).content.strip()
            
            if relevance_response.lower() != 'yes':
                return "Please ask a question related to the CV or the analysis."

            # Answer the question
            answer_prompt = [
                SystemMessage(content="You are an expert in CV analysis and recruitment. Answer the user's question based on the provided information. Use the analysis results for questions about ranking, scores, or comparisons. Use the CV text for questions about specific details in the CV. If the information is not available, say so. Provide a concise and accurate response."),
                HumanMessage(content=f"Job Role: {job_role}\n\nAnalysis Results (JSON):\n{analysis_json}\n\nSelected CV: {selected_cv}\n\nCV Text:\n{cv_text}\n\nQuestion: {question}")
            ]
            response = llm.invoke(answer_prompt)
            return response.content.strip()
        except Exception as e:
            logging.error(f"Error processing chatbot query: {str(e)}")
            return "Sorry, an error occurred while processing your question."

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

    # CV Selection for Chatbot
    tk.Label(root, text="Select CV for Chatbot:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    cv_var = tk.StringVar()
    cv_dropdown = ttk.OptionMenu(root, cv_var, "Select CV after analysis")
    cv_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    # Results Area
    result_text = scrolledtext.ScrolledText(root, width=90, height=15, state='normal')
    result_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

    # Chatbot Area
    tk.Label(root, text="Chatbot: Ask about the selected CV or the analysis").grid(row=5, column=0, padx=5, pady=5, sticky="w")
    chat_text = scrolledtext.ScrolledText(root, width=90, height=10, state='disabled')
    chat_text.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

    tk.Label(root, text="Your Question:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
    question_entry = tk.Entry(root, width=60)
    question_entry.grid(row=7, column=1, padx=5, pady=5)

    # Redirect logging to result text area
    text_handler = TextHandler(result_text)
    text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().handlers = [text_handler]

    # Store CV texts and ranked CVs
    cv_texts = {}
    ranked_cvs = []

    # Run Analysis
    def run_analysis():
        nonlocal cv_texts, ranked_cvs
        cv_folder = folder_entry.get()
        job_role = job_role_var.get()
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.config(state='disabled')
        if not cv_folder or not job_role:
            logging.error("Please select a CV folder and job role.")
            return
        try:
            ranked_cvs, cv_texts = rank_cvs(cv_folder, job_role)
            result_text.config(state='normal')
            result_text.insert(tk.END, "Ranked CVs:\n")
            for i, cv in enumerate(ranked_cvs, 1):
                result_text.insert(tk.END, f"Rank {i}: {cv['cv_file']}\n")
                result_text.insert(tk.END, f"  Years of Experience: {cv['years_experience']} (Weight: {cv['experience_weight']})\n")
                result_text.insert(tk.END, f"  Skill Match: {cv['skill_match_percent']}%\n")
                result_text.insert(tk.END, f"  CV Skills: {cv['cv_skills']}\n")
                result_text.insert(tk.END, f"  Missing Skills: {cv['missing_skills']}\n")
                result_text.insert(tk.END, f"  Skill Score: {cv['skill_score']}\n")
                result_text.insert(tk.END, f"  Total Score: {cv['total_score']}\n\n")
            result_text.config(state='disabled')

            # Update CV dropdown
            cv_files = [cv['cv_file'] for cv in ranked_cvs]
            if cv_files:
                cv_var.set(cv_files[0])
            else:
                cv_var.set("No CVs available")
            menu = cv_dropdown["menu"]
            menu.delete(0, "end")
            for cv_file in cv_files:
                menu.add_command(label=cv_file, command=lambda value=cv_file: cv_var.set(value))
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")

    # Chatbot Send Button
    def send_question():
        if not llm:
            chat_text.config(state='normal')
            chat_text.insert(tk.END, "Error: Chatbot not initialized.\n")
            chat_text.config(state='disabled')
            chat_text.yview(tk.END)
            return

        question = question_entry.get().strip()
        selected_cv = cv_var.get()
        if not question or selected_cv not in cv_texts:
            chat_text.config(state='normal')
            chat_text.insert(tk.END, "Please select a CV and enter a question.\n")
            chat_text.config(state='disabled')
            chat_text.yview(tk.END)
            return

        cv_text = cv_texts[selected_cv]
        response = ask_cv_question(cv_text, question, selected_cv)
        
        chat_text.config(state='normal')
        chat_text.insert(tk.END, f"You: {question}\n")
        chat_text.insert(tk.END, f"Chatbot: {response}\n\n")
        chat_text.config(state='disabled')
        chat_text.yview(tk.END)
        question_entry.delete(0, tk.END)

    tk.Button(root, text="Run Analysis", command=run_analysis).grid(row=3, column=1, padx=5, pady=5)
    tk.Button(root, text="Send", command=send_question).grid(row=7, column=2, padx=5, pady=5)

    root.mainloop()

# Run the app
if __name__ == "__main__":
    create_gui()