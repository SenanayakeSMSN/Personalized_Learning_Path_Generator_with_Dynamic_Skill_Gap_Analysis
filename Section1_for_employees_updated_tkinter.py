'''
# Cell 1: Install Dependencies
#%pip install pdfplumber pymongo google-generativeai dnspython pytesseract Pillow tkinter

# Cell 2: Import Libraries and Setup
import os
import json
import logging
import re
from pymongo import MongoClient
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import google.generativeai as genai
import pdfplumber
import pytesseract
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcode credentials (replace with your key; use environment variables in production)
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo"  # Replace with your Gemini API key
MONGODB_URI = "mongodb+srv://shashi:VSXV9WDNmRvYnA7p@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority"

# Verify dependencies
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
    from PIL import Image
except ImportError:
    raise ImportError("pytesseract or PIL is not installed. Run: %pip install pytesseract Pillow")

# Cell 3: MongoDB Connection
def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
        collection.create_index("Job_Role")
        logging.info("Connected to MongoDB Atlas")
        return collection
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

def get_job_roles(collection):
    try:
        job_roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1}) if "Job_Role" in doc]
        job_roles = sorted(list(set(job_roles)))  # Remove duplicates and sort
        logging.info(f"Retrieved job roles: {job_roles}")
        return job_roles
    except Exception as e:
        logging.error(f"Error retrieving job roles: {str(e)}")
        return []

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

# Cell 5: Gemini Skill Extraction
def extract_skills_with_gemini(cv_text):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert in CV analysis. Perform the following steps on the provided CV text:\n"
            "1. Parse the CV text into a structured JSON object with the following fields (include only fields that can be identified, leave empty if not found):\n"
            "   - name: The candidate's full name (string).\n"
            "   - education: List of education entries (array of strings, e.g., ['B.S. Computer Science, XYZ University, 2020']).\n"
            "   - projects: List of project descriptions (array of strings, e.g., ['Built a web app using Django']).\n"
            "   - experience: List of work experience entries (array of strings, e.g., ['Software Engineer at ABC Corp, 2021-2023']).\n"
            "   - skills: List of explicit skills from a 'Skills' or similar section (array of strings, e.g., ['Python', 'Java']).\n"
            "   - contact: Contact information (string, e.g., 'email: john@example.com').\n"
            "2. From the structured JSON, extract all skills into a separate JSON array, including:\n"
            "   - Explicit skills listed in the 'skills' field (e.g., 'Python, TensorFlow').\n"
            "   - Implicit skills inferred from 'projects', 'experience', or other sections (e.g., 'Built a fraud detection model using scikit-learn' implies 'scikit-learn').\n"
            "   - Combine similar skills (e.g., 'ML' and 'Machine Learning' as 'Machine Learning').\n"
            "   - Capitalize skills appropriately (e.g., 'PyTorch', 'AWS SageMaker').\n"
            "   - Return [] if no skills are found or input is empty.\n"
            "Return a JSON object with two fields:\n"
            "   - structured_cv: The structured JSON object from step 1.\n"
            "   - skills: The JSON array of distinct skills from step 2.\n"
            "Output must be a valid JSON object without ```json, backticks, or any other formatting.\n"
            "Example:\n"
            "Input: 'Name: John Doe\nSkills: Python, ML\nProjects: Built a model using TensorFlow'\n"
            "Output: {\"structured_cv\":{\"name\":\"John Doe\",\"skills\":[\"Python\",\"ML\"],\"projects\":[\"Built a model using TensorFlow\"],\"education\":[],\"experience\":[],\"contact\":\"\"},\"skills\":[\"Python\",\"Machine Learning\",\"TensorFlow\"]}"
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
            result = json.loads(cleaned_response)
            if not isinstance(result, dict) or "structured_cv" not in result or "skills" not in result:
                logging.error("Gemini response is not a valid JSON object with required fields")
                return [], {}
            structured_cv = result["structured_cv"]
            skills = result["skills"]
            if not isinstance(skills, list):
                logging.error("Gemini skills field is not a JSON array")
                return [], structured_cv
            logging.info(f"Structured CV: {json.dumps(structured_cv, indent=2)}")
            logging.info(f"Extracted skills: {skills}")
            return skills, structured_cv
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse cleaned Gemini response as JSON: {cleaned_response}, Error: {str(e)}")
            return [], {}
    except Exception as e:
        logging.error(f"Error in Gemini skill extraction: {str(e)}")
        return [], {}

# Cell 6: Gemini Custom Question Answering
def answer_cv_question(cv_text, structured_cv, question, skill_gap_result=None):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        context = f"CV Text:\n{cv_text}\n\nStructured CV (parsed data):\n{json.dumps(structured_cv, indent=2)}\n\n"
        if skill_gap_result:
            context += f"Skill Gap Analysis:\nJob Role: {skill_gap_result.get('job_role', '')}\nCV Skills: {', '.join(skill_gap_result.get('cv_skills', []))}\nRequired Skills: {', '.join(skill_gap_result.get('required_skills', []))}\nMissing Skills: {', '.join(skill_gap_result.get('missing_skills', []))}\n\n"
        prompt = (
            "You are an expert career advisor. Using the provided CV text, structured CV data, and skill gap analysis (if available), answer the following question in a clear, personalized, and professional manner. Provide specific recommendations if the question involves areas for improvement, courses, or career paths. If asked about a learning path, suggest a step-by-step plan with skills to learn, recommended courses (e.g., from Coursera, Udemy), and a timeline. Return the answer as plain text.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logging.info(f"Gemini answered question '{question}': {answer[:100]}...")
        return answer
    except Exception as e:
        logging.error(f"Error answering question with Gemini: {str(e)}")
        return f"Error answering question: {str(e)}"

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

# Cell 8: Skill Gap Analysis
def skill_gap_analysis(cv_skills, required_skills):
    try:
        cv_skills_set = set(skill.lower() for skill in cv_skills)
        required_skills_set = set(skill.lower() for skill in required_skills)
        missing_skills = required_skills_set - cv_skills_set
        result = [skill for skill in required_skills if skill.lower() in missing_skills]
        logging.info(f"Missing skills: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        raise ValueError(f"Error in skill gap analysis: {str(e)}")

# Cell 9: Tkinter GUI
class SkillGapAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skill Gap Analyzer")
        self.root.geometry("800x600")
        self.job_collection = connect_to_mongodb()
        self.job_roles = get_job_roles(self.job_collection)
        self.cv_path = None
        self.cv_text = None
        self.structured_cv = None
        self.skill_gap_result = None

        # GUI Components
        self.frame = tk.Frame(self.root, padx=10, pady=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Status Label
        tk.Label(self.frame, text="Current CV:").grid(row=0, column=0, sticky="w")
        self.status_label = tk.Label(self.frame, text="No CV uploaded", width=50)
        self.status_label.grid(row=0, column=1, sticky="w")
        tk.Button(self.frame, text="Clear CV", command=self.clear_cv).grid(row=0, column=2)

        # CV Upload
        tk.Label(self.frame, text="Upload CV (PDF):").grid(row=1, column=0, sticky="w")
        self.cv_label = tk.Label(self.frame, text="No file selected", width=50)
        self.cv_label.grid(row=1, column=1, sticky="w")
        tk.Button(self.frame, text="Browse", command=self.upload_cv).grid(row=1, column=2)

        # Job Role Selection (Dropdown)
        tk.Label(self.frame, text="Target Job Role:").grid(row=2, column=0, sticky="w")
        self.job_role_var = tk.StringVar()
        if self.job_roles:
            self.job_role_dropdown = ttk.Combobox(self.frame, textvariable=self.job_role_var, values=self.job_roles, width=47, state="readonly")
            self.job_role_dropdown.grid(row=2, column=1, sticky="w")
        else:
            self.job_role_dropdown = tk.Entry(self.frame, textvariable=self.job_role_var, width=50)
            self.job_role_dropdown.grid(row=2, column=1, sticky="w")
            logging.warning("No job roles found in MongoDB. Using text entry for job role.")
        tk.Button(self.frame, text="Analyze", command=self.analyze_cv).grid(row=2, column=2)

        # Custom Question
        tk.Label(self.frame, text="Ask a Question:").grid(row=3, column=0, sticky="w")
        self.question_entry = tk.Entry(self.frame, width=50)
        self.question_entry.grid(row=3, column=1, sticky="w")
        tk.Button(self.frame, text="Submit Question", command=self.ask_question).grid(row=3, column=2)

        # Generate Learning Path
        tk.Button(self.frame, text="Generate Learning Path", command=self.generate_learning_path).grid(row=4, column=1, pady=5)

        # Results Display
        tk.Label(self.frame, text="Results:").grid(row=5, column=0, sticky="nw")
        self.result_text = tk.Text(self.frame, height=20, width=70, wrap=tk.WORD)
        self.result_text.grid(row=5, column=1, columnspan=2, pady=5)
        self.scrollbar = tk.Scrollbar(self.frame, command=self.result_text.yview)
        self.scrollbar.grid(row=5, column=3, sticky="ns")
        self.result_text.config(yscrollcommand=self.scrollbar.set)

    def upload_cv(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.cv_path = file_path
            self.cv_label.config(text=os.path.basename(file_path))
            self.status_label.config(text=os.path.basename(file_path))
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "CV uploaded. Select a job role and click 'Analyze'.\n")
            self.cv_text = None
            self.structured_cv = None
            self.skill_gap_result = None
            logging.info(f"Selected CV: {file_path}")

    def clear_cv(self):
        self.cv_path = None
        self.cv_text = None
        self.structured_cv = None
        self.skill_gap_result = None
        self.cv_label.config(text="No file selected")
        self.status_label.config(text="No CV uploaded")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "CV cleared. Upload a new CV to continue.\n")
        logging.info("Cleared CV data")

    def analyze_cv(self):
        if not self.cv_path:
            messagebox.showerror("Error", "Please upload a CV.")
            return
        job_role = self.job_role_var.get().strip()
        if not job_role:
            messagebox.showerror("Error", "Please select or enter a target job role.")
            return
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Analyzing CV...\n")
        self.root.update()
        try:
            self.cv_text = extract_cv_text(self.cv_path)
            cv_skills, structured_cv = extract_skills_with_gemini(self.cv_text)
            self.structured_cv = structured_cv
            if not cv_skills:
                messagebox.showerror("Error", "Failed to extract skills from CV.")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Failed to extract skills from CV.\n")
                return
            required_skills = get_required_skills(self.job_collection, job_role)
            missing_skills = skill_gap_analysis(cv_skills, required_skills)
            self.skill_gap_result = {
                "job_role": job_role,
                "cv_skills": cv_skills,
                "required_skills": required_skills,
                "missing_skills": missing_skills
            }
            result_text = (
                f"Skill Gap Analysis:\n"
                f"Job Role: {job_role}\n"
                f"CV Skills: {', '.join(cv_skills)}\n"
                f"Required Skills: {', '.join(required_skills)}\n"
                f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}\n"
            )
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            logging.info(f"Skill gap analysis completed for {job_role}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}\n")
            logging.error(f"Analysis error: {str(e)}")

    def ask_question(self):
        if not self.cv_text or not self.structured_cv:
            messagebox.showerror("Error", "Please analyze a CV first.")
            return
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question.")
            return
        self.result_text.insert(tk.END, f"\nQuestion: {question}\n")
        self.result_text.insert(tk.END, "Processing...\n")
        self.root.update()
        try:
            answer = answer_cv_question(self.cv_text, self.structured_cv, question, self.skill_gap_result)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Question: {question}\nAnswer: {answer}\n")
            self.question_entry.delete(0, tk.END)  # Clear question input
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_text.insert(tk.END, f"Error answering question: {str(e)}\n")

    def generate_learning_path(self):
        if not self.cv_text or not self.structured_cv or not self.skill_gap_result:
            messagebox.showerror("Error", "Please analyze a CV first.")
            return
        question = f"Generate a learning path to acquire the missing skills ({', '.join(self.skill_gap_result.get('missing_skills', []))}) for the job role '{self.skill_gap_result.get('job_role', '')}'. Include specific courses, resources, and a timeline."
        self.result_text.insert(tk.END, f"\nGenerating Learning Path...\n")
        self.root.update()
        try:
            learning_path = answer_cv_question(self.cv_text, self.structured_cv, question, self.skill_gap_result)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Learning Path for {self.skill_gap_result.get('job_role', '')}:\n{learning_path}\n")
            self.question_entry.delete(0, tk.END)  # Clear question input
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_text.insert(tk.END, f"Error generating learning path: {str(e)}\n")

# Cell 10: Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = SkillGapAnalyzerApp(root)
    root.mainloop()
'''

# Cell 1: Install Dependencies
#%pip install pdfplumber pymongo google-generativeai dnspython pytesseract Pillow tkinter

# Cell 2: Import Libraries and Setup
import os
import json
import logging
import re
from pymongo import MongoClient
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import google.generativeai as genai
import pdfplumber
import pytesseract
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcode credentials (replace with your key; use environment variables in production)
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo"  # Replace with your Gemini API key
MONGODB_URI = "mongodb+srv://shashi:VSXV9WDNmRvYnA7p@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority"

# Verify dependencies
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
    from PIL import Image
except ImportError:
    raise ImportError("pytesseract or PIL is not installed. Run: %pip install pytesseract Pillow")

# Cell 3: MongoDB Connection
def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
        collection.create_index("Job_Role")
        logging.info("Connected to MongoDB Atlas")
        return collection
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

def get_job_roles(collection):
    try:
        job_roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1}) if "Job_Role" in doc]
        job_roles = sorted(list(set(job_roles)))  # Remove duplicates and sort
        logging.info(f"Retrieved job roles: {job_roles}")
        return job_roles
    except Exception as e:
        logging.error(f"Error retrieving job roles: {str(e)}")
        return []

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

# Cell 5: Gemini Skill and Experience Extraction
def extract_skills_with_gemini(cv_text):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an expert in CV analysis. Perform the following steps on the provided CV text:\n"
            "1. Parse the CV text into a structured JSON object with the following fields (include only fields that can be identified, leave empty if not found):\n"
            "   - name: The candidate's full name (string).\n"
            "   - education: List of education entries (array of strings, e.g., ['B.S. Computer Science, XYZ University, 2020']).\n"
            "   - projects: List of project descriptions (array of strings, e.g., ['Built a web app using Django']).\n"
            "   - experience: List of work experience entries with dates (array of strings, e.g., ['Software Engineer at ABC Corp, Jan 2019 - Dec 2021']).\n"
            "   - skills: List of explicit skills from a 'Skills' or similar section (array of strings, e.g., ['Python', 'Java']).\n"
            "   - contact: Contact information (string, e.g., 'email: john@example.com').\n"
            "   - experience_duration: Total work experience duration as a string, calculated as follows:\n"
            "     - Identify the start date of the first appointed job and the end date of the last appointed job (use 'Present' for ongoing roles, assuming today's date).\n"
            "     - Calculate the total duration in years and months (e.g., '2 years 3 months', '6 months', '3 years', or 'No experience' if no experience is found).\n"
            "     - Handle partial date formats (e.g., '2019 - 2021' as 'Jan 2019 - Dec 2021') by assuming start of first month and end of last month.\n"
            "2. Extract all skills into a separate JSON array, including:\n"
            "   - Explicit skills listed in the 'skills' field (e.g., 'Python, TensorFlow').\n"
            "   - Implicit skills inferred from 'projects', 'experience', or other sections (e.g., 'Built a fraud detection model using scikit-learn' implies 'scikit-learn').\n"
            "   - Combine similar skills (e.g., 'ML' and 'Machine Learning' as 'Machine Learning').\n"
            "   - Capitalize skills appropriately (e.g., 'PyTorch', 'AWS SageMaker').\n"
            "   - Return [] if no skills are found or input is empty.\n"
            "Return a JSON object with three fields:\n"
            "   - structured_cv: The structured JSON object from step 1.\n"
            "   - skills: The JSON array of distinct skills from step 2.\n"
            "   - experience_duration: The total work experience duration from step 1 (redundant for compatibility).\n"
            "Output must be a valid JSON object without ```json, backticks, or any other formatting.\n"
            "Example:\n"
            "Input: 'Name: John Doe\nSkills: Python, ML\nExperience: Software Engineer at ABC Corp, Jan 2019 - Dec 2021; Data Scientist at XYZ Inc, Mar 2022 - Present'\n"
            "Output: {\"structured_cv\":{\"name\":\"John Doe\",\"skills\":[\"Python\",\"ML\"],\"experience\":[\"Software Engineer at ABC Corp, Jan 2019 - Dec 2021\",\"Data Scientist at XYZ Inc, Mar 2022 - Present\"],\"projects\":[],\"education\":[],\"contact\":\"\",\"experience_duration\":\"3 years 3 months\"},\"skills\":[\"Python\",\"Machine Learning\"],\"experience_duration\":\"3 years 3 months\"}"
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
            result = json.loads(cleaned_response)
            if not isinstance(result, dict) or "structured_cv" not in result or "skills" not in result or "experience_duration" not in result:
                logging.error("Gemini response is not a valid JSON object with required fields")
                return [], {}, "No experience"
            structured_cv = result["structured_cv"]
            skills = result["skills"]
            experience_duration = result["experience_duration"]
            if not isinstance(skills, list):
                logging.error("Gemini skills field is not a JSON array")
                return [], structured_cv, "No experience"
            if not isinstance(experience_duration, str):
                logging.error("Gemini experience_duration field is not a string")
                return [], structured_cv, "No experience"
            logging.info(f"Structured CV: {json.dumps(structured_cv, indent=2)}")
            logging.info(f"Extracted skills: {skills}")
            logging.info(f"Extracted experience duration: {experience_duration}")
            return skills, structured_cv, experience_duration
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse cleaned Gemini response as JSON: {cleaned_response}, Error: {str(e)}")
            return [], {}, "No experience"
    except Exception as e:
        logging.error(f"Error in Gemini skill extraction: {str(e)}")
        return [], {}, "No experience"

# Cell 6: Gemini Custom Question Answering
def answer_cv_question(cv_text, structured_cv, question, skill_gap_result=None, experience_duration="No experience"):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        context = f"CV Text:\n{cv_text}\n\nStructured CV (parsed data):\n{json.dumps(structured_cv, indent=2)}\n\nTotal Work Experience: {experience_duration}\n\n"
        if skill_gap_result:
            context += f"Skill Gap Analysis:\nJob Role: {skill_gap_result.get('job_role', '')}\nCV Skills: {', '.join(skill_gap_result.get('cv_skills', []))}\nRequired Skills: {', '.join(skill_gap_result.get('required_skills', []))}\nMissing Skills: {', '.join(skill_gap_result.get('missing_skills', []))}\n\n"
        prompt = (
            "You are an expert career advisor. Using the provided CV text, structured CV data, total work experience, and skill gap analysis (if available), answer the following question in a clear, personalized, and professional manner. Provide specific recommendations if the question involves areas for improvement, courses, or career paths. If asked about a learning path, suggest a step-by-step plan with skills to learn, recommended courses (e.g., from Coursera, Udemy), and a timeline. Return the answer as plain text.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logging.info(f"Gemini answered question '{question}': {answer[:100]}...")
        return answer
    except Exception as e:
        logging.error(f"Error answering question with Gemini: {str(e)}")
        return f"Error answering question: {str(e)}"

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

# Cell 8: Skill Gap Analysis
def skill_gap_analysis(cv_skills, required_skills):
    try:
        cv_skills_set = set(skill.lower() for skill in cv_skills)
        required_skills_set = set(skill.lower() for skill in required_skills)
        missing_skills = required_skills_set - cv_skills_set
        result = [skill for skill in required_skills if skill.lower() in missing_skills]
        logging.info(f"Missing skills: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        raise ValueError(f"Error in skill gap analysis: {str(e)}")

# Cell 9: Tkinter GUI
class SkillGapAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skill Gap Analyzer")
        self.root.geometry("800x600")
        self.job_collection = connect_to_mongodb()
        self.job_roles = get_job_roles(self.job_collection)
        self.cv_path = None
        self.cv_text = None
        self.structured_cv = None
        self.skill_gap_result = None

        # GUI Components
        self.frame = tk.Frame(self.root, padx=10, pady=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Status Label
        tk.Label(self.frame, text="Current CV:").grid(row=0, column=0, sticky="w")
        self.status_label = tk.Label(self.frame, text="No CV uploaded", width=50)
        self.status_label.grid(row=0, column=1, sticky="w")
        tk.Button(self.frame, text="Clear CV", command=self.clear_cv).grid(row=0, column=2)

        # CV Upload
        tk.Label(self.frame, text="Upload CV (PDF):").grid(row=1, column=0, sticky="w")
        self.cv_label = tk.Label(self.frame, text="No file selected", width=50)
        self.cv_label.grid(row=1, column=1, sticky="w")
        tk.Button(self.frame, text="Browse", command=self.upload_cv).grid(row=1, column=2)

        # Job Role Selection (Dropdown)
        tk.Label(self.frame, text="Target Job Role:").grid(row=2, column=0, sticky="w")
        self.job_role_var = tk.StringVar()
        if self.job_roles:
            self.job_role_dropdown = ttk.Combobox(self.frame, textvariable=self.job_role_var, values=self.job_roles, width=47, state="readonly")
            self.job_role_dropdown.grid(row=2, column=1, sticky="w")
        else:
            self.job_role_dropdown = tk.Entry(self.frame, textvariable=self.job_role_var, width=50)
            self.job_role_dropdown.grid(row=2, column=1, sticky="w")
            logging.warning("No job roles found in MongoDB. Using text entry for job role.")
        tk.Button(self.frame, text="Analyze", command=self.analyze_cv).grid(row=2, column=2)

        # Custom Question
        tk.Label(self.frame, text="Ask a Question:").grid(row=3, column=0, sticky="w")
        self.question_entry = tk.Entry(self.frame, width=50)
        self.question_entry.grid(row=3, column=1, sticky="w")
        tk.Button(self.frame, text="Submit Question", command=self.ask_question).grid(row=3, column=2)

        # Generate Learning Path
        tk.Button(self.frame, text="Generate Learning Path", command=self.generate_learning_path).grid(row=4, column=1, pady=5)

        # Results Display
        tk.Label(self.frame, text="Results:").grid(row=5, column=0, sticky="nw")
        self.result_text = tk.Text(self.frame, height=20, width=70, wrap=tk.WORD)
        self.result_text.grid(row=5, column=1, columnspan=2, pady=5)
        self.scrollbar = tk.Scrollbar(self.frame, command=self.result_text.yview)
        self.scrollbar.grid(row=5, column=3, sticky="ns")
        self.result_text.config(yscrollcommand=self.scrollbar.set)

    def upload_cv(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.cv_path = file_path
            self.cv_label.config(text=os.path.basename(file_path))
            self.status_label.config(text=os.path.basename(file_path))
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "CV uploaded. Select a job role and click 'Analyze'.\n")
            self.cv_text = None
            self.structured_cv = None
            self.skill_gap_result = None
            logging.info(f"Selected CV: {file_path}")

    def clear_cv(self):
        self.cv_path = None
        self.cv_text = None
        self.structured_cv = None
        self.skill_gap_result = None
        self.cv_label.config(text="No file selected")
        self.status_label.config(text="No CV uploaded")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "CV cleared. Upload a new CV to continue.\n")
        logging.info("Cleared CV data")

    def analyze_cv(self):
        if not self.cv_path:
            messagebox.showerror("Error", "Please upload a CV.")
            return
        job_role = self.job_role_var.get().strip()
        if not job_role:
            messagebox.showerror("Error", "Please select or enter a target job role.")
            return
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Analyzing CV...\n")
        self.root.update()
        try:
            self.cv_text = extract_cv_text(self.cv_path)
            cv_skills, structured_cv, experience_duration = extract_skills_with_gemini(self.cv_text)
            self.structured_cv = structured_cv
            if not cv_skills:
                messagebox.showerror("Error", "Failed to extract skills from CV.")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Failed to extract skills from CV.\n")
                return
            required_skills = get_required_skills(self.job_collection, job_role)
            missing_skills = skill_gap_analysis(cv_skills, required_skills)
            self.skill_gap_result = {
                "job_role": job_role,
                "cv_skills": cv_skills,
                "required_skills": required_skills,
                "missing_skills": missing_skills,
                "experience_duration": experience_duration
            }
            result_text = (
                f"Skill Gap Analysis:\n"
                f"Job Role: {job_role}\n"
                f"Total Work Experience: {experience_duration}\n"
                f"CV Skills: {', '.join(cv_skills)}\n"
                f"Required Skills: {', '.join(required_skills)}\n"
                f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}\n"
            )
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            logging.info(f"Skill gap analysis completed for {job_role}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}\n")
            logging.error(f"Analysis error: {str(e)}")

    def ask_question(self):
        if not self.cv_text or not self.structured_cv:
            messagebox.showerror("Error", "Please analyze a CV first.")
            return
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question.")
            return
        self.result_text.insert(tk.END, f"\nQuestion: {question}\n")
        self.result_text.insert(tk.END, "Processing...\n")
        self.root.update()
        try:
            answer = answer_cv_question(
                self.cv_text,
                self.structured_cv,
                question,
                self.skill_gap_result,
                self.skill_gap_result.get("experience_duration", "No experience")
            )
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Question: {question}\nAnswer: {answer}\n")
            self.question_entry.delete(0, tk.END)  # Clear question input
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_text.insert(tk.END, f"Error answering question: {str(e)}\n")

    def generate_learning_path(self):
        if not self.cv_text or not self.structured_cv or not self.skill_gap_result:
            messagebox.showerror("Error", "Please analyze a CV first.")
            return
        question = f"Generate a learning path to acquire the missing skills ({', '.join(self.skill_gap_result.get('missing_skills', []))}) for the job role '{self.skill_gap_result.get('job_role', '')}', considering the candidate has {self.skill_gap_result.get('experience_duration', 'No experience')} of work experience. Include specific courses, resources, and a timeline."
        self.result_text.insert(tk.END, f"\nGenerating Learning Path...\n")
        self.root.update()
        try:
            learning_path = answer_cv_question(
                self.cv_text,
                self.structured_cv,
                question,
                self.skill_gap_result,
                self.skill_gap_result.get("experience_duration", "No experience")
            )
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Learning Path for {self.skill_gap_result.get('job_role', '')}:\n{learning_path}\n")
            self.question_entry.delete(0, tk.END)  # Clear question input
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result_text.insert(tk.END, f"Error generating learning path: {str(e)}\n")

# Cell 10: Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = SkillGapAnalyzerApp(root)
    root.mainloop()