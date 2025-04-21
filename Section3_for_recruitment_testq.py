
# Cell 1: Placeholder (No code)

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
# --- SECURITY WARNING: Do not hardcode API keys in production code. Use environment variables or secret management. ---
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo"  # Replace with your actual Gemini API key
MONGODB_URI = "mongodb+srv://akilapremarathna0:123@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority" # Replace with your MongoDB URI
# --- End Security Warning ---

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
    # Example paths:
    # Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # macOS/Linux: Usually in PATH, but if not: '/usr/local/bin/tesseract' or similar
    # Check your Tesseract installation path and uncomment/update the line below if needed.
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    raise ImportError("pytesseract or PIL is not installed. Run: %pip install pytesseract Pillow")
except Exception as e:
    # Catch potential errors if Tesseract is not found even if pytesseract is installed
    logging.warning(f"Could not configure pytesseract path: {e}. Ensure Tesseract is installed and in your PATH or set the path manually.")


from langchain_core.messages import HumanMessage, SystemMessage

# Cell 3: MongoDB Connection
def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        # Ping the server to confirm connection
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
        # Ensure index exists for faster lookups
        collection.create_index("Job_Role")
        job_roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1})]
        if not job_roles:
            logging.warning("No job roles found in the database. Check the 'jobrole_skill' collection.")
        logging.info(f"Connected to MongoDB Atlas. Found job roles: {job_roles}")
        return collection, job_roles
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

# Cell 4: Enhanced CV Text Extraction
def extract_cv_text(cv_path):
    text = ""
    try:
        with pdfplumber.open(cv_path) as pdf:
            logging.info(f"Processing PDF: {cv_path}")
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n\n"
                    logging.info(f"Extracted text from page {page_number} using pdfplumber")
                else:
                    # Try OCR if text extraction yields little or no text
                    logging.warning(f"No significant text extracted from page {page_number} via standard method. Attempting OCR.")
                    try:
                        # Increase resolution for potentially better OCR results
                        img = page.to_image(resolution=300).original
                        # Specify language if known, e.g., lang='eng'
                        ocr_text = pytesseract.image_to_string(img, lang='eng')
                        if ocr_text and ocr_text.strip():
                            text += ocr_text + "\n\n"
                            logging.info(f"Extracted OCR text from page {page_number}")
                        else:
                            logging.warning(f"OCR yielded no significant text for page {page_number}")
                            text += "\n\n" # Add space even if empty
                    except Exception as ocr_e:
                        logging.error(f"OCR failed for page {page_number}: {str(ocr_e)}. Check Tesseract installation and path.")
                        text += "\n\n" # Add space to maintain structure
            if not text.strip():
                raise ValueError("No text extracted from CV after attempting both standard extraction and OCR. Check PDF content/format or Tesseract installation.")
            logging.info(f"Successfully extracted text from CV: {text[:200]}...") # Log more text
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error(f"PDF file '{os.path.basename(cv_path)}' is password-protected. Cannot process.")
        raise ValueError("PDF is password-protected")
    except FileNotFoundError:
        logging.error(f"CV file not found at path: {cv_path}")
        raise ValueError(f"CV file not found: {cv_path}")
    except Exception as e:
        logging.error(f"Error reading CV '{os.path.basename(cv_path)}': {str(e)}")
        # Raise a more specific error or return None/empty string depending on desired behavior
        raise ValueError(f"Error processing CV: {str(e)}")


# Cell 5: Skills Section Extraction (Optional Enhancement - Not used by Gemini in this version)
# Note: The current Gemini prompt analyzes the *entire* CV text for skills, making this function less critical
# unless you want to pre-filter or focus the analysis later.
def extract_skills_section(cv_text):
    try:
        # Improved regex to handle variations and stop words better
        match = re.search(
            r"(?i)\b(skills|technical\s+skills|key\s+skills|core\s+competencies|technical\s+expertise|proficiencies)\b[:\s]*\n?(.*?)(?=\n\s*\b(experience|work\s+history|employment|education|projects|publications|awards|certifications|references|contact|personal\s+details)\b|\Z)",
            cv_text, re.DOTALL
        )
        if match:
            skills_text = match.group(2).strip()
            # Further clean up potential list markers and condense whitespace
            skills_text = re.sub(r'^\s*[-*•]\s*', '', skills_text, flags=re.MULTILINE) # Remove list bullets
            skills_text = re.sub(r'\s+', ' ', skills_text) # Condense whitespace
            if skills_text:
                 logging.info(f"Extracted potential skills section: {skills_text[:100]}...")
                 return skills_text
            else:
                 logging.warning("Found skills section header but no content extracted.")
                 return cv_text # Fallback to full text
        logging.warning("No explicit skills section found using regex patterns. Using full CV text for Gemini.")
        return cv_text
    except Exception as e:
        logging.error(f"Error during regex extraction of skills section: {str(e)}")
        return cv_text # Fallback to full text

# Cell 6: MODIFIED - Combined function to extract both skills and *role-specific* experience
def extract_cv_data_with_gemini(cv_text, job_role): # Added job_role parameter
    """
    Extracts skills and calculates relevant years of experience using Gemini,
    tailored to a specific job role.
    """
    if not cv_text or not cv_text.strip():
        logging.error("CV text is empty, cannot extract data.")
        return [], 0
    if not job_role or not job_role.strip():
        logging.error("Job role is empty, cannot calculate relevant experience.")
        return [], 0

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Consider using a more robust model if 'flash' has issues with complex instructions or long CVs
        model = genai.GenerativeModel("gemini-1.5-flash")

        # --- Using the new prompt format ---
        current_year = datetime.now().year
        current_month = datetime.now().month
        prompt = (
            "You are an expert in CV analysis for the job role of '" + job_role + "'. Extract the following from the CV text:\n\n"
            "1. SKILLS: Extract all skills including explicit skills (from 'Skills' sections) and implicit skills (inferred from experience/projects).\n"
            "2. EXPERIENCE: Calculate the total years of professional experience relevant to the '" + job_role + "' role by analyzing job entries and their date ranges. Only include experience from roles that are directly related to '" + job_role + "'.\n\n"
            "For SKILLS:\n"
            "- Return distinct skills (e.g., capitalize 'PyTorch', 'AWS SageMaker').\n"
            "- Combine similar skills (e.g., 'ML' and 'Machine Learning' as 'Machine Learning', 'Gen AI' and 'Generative AI' as 'Generative AI').\n\n"
            "For EXPERIENCE:\n"
            "- Identify job entries that are relevant to the '" + job_role + "' role. Consider job titles, descriptions, and skills used.\n"
            "- For each relevant job entry, calculate the duration in months: (end_year - start_year) * 12 + (end_month - start_month).\n"
            f"- If end date is 'Present', 'Current', or similar, use {current_month}/{current_year} as the end date.\n"
            "- For ambiguous dates (only year mentioned), assume January for the start month and December for the end month. If only start year, assume Jan. If only end year, assume Dec.\n"
            "- Sum the durations (in months) of all relevant job entries.\n"
            "- Convert the total months to years by dividing by 12. Round the result to the nearest integer.\n\n"
            "Return a JSON object with exactly two keys:\n"
            "- 'skills': array of unique skill strings (string[])\n"
            "- 'years': integer representing total years of relevant experience (integer)\n\n"
            "Example output for a Data Scientist role: {\"skills\": [\"Python\", \"Machine Learning\", \"TensorFlow\", \"SQL\"], \"years\": 5}\n\n"
            "Your output MUST be a single, valid JSON object containing only the 'skills' and 'years' keys, without any surrounding text, comments, or markdown formatting like ```json ... ```.\n\n"
            "CV Text:\n" + cv_text
        )
        # --- End of new prompt format ---

        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw response for CV data (first 100 chars): {raw_response[:100]}...")

        # Clean response - remove markdown and potential extra text
        cleaned_response = re.sub(r"^```json\s*", "", raw_response) # Remove starting ```json
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response) # Remove ending ```
        cleaned_response = cleaned_response.strip() # Remove leading/trailing whitespace

        try:
            cv_data = json.loads(cleaned_response)
            if not isinstance(cv_data, dict):
                 raise ValueError("Parsed JSON is not a dictionary.")
            if "skills" not in cv_data or "years" not in cv_data:
                logging.error(f"Gemini response missing required keys ('skills', 'years'). Response: {cleaned_response}")
                # Attempt to find skills/years even if format is wrong (basic fallback)
                skills_fallback = re.findall(r'"([^"]+)"', cleaned_response) # Very basic skill extraction
                years_match = re.search(r'"years":\s*(\d+)', cleaned_response)
                years_fallback = int(years_match.group(1)) if years_match else 0
                return skills_fallback, years_fallback

            skills = cv_data.get("skills", [])
            years = cv_data.get("years", 0)

            # Validate types
            if not isinstance(skills, list):
                logging.warning(f"Expected 'skills' to be a list, got {type(skills)}. Attempting conversion or returning empty.")
                skills = list(skills) if isinstance(skills, (tuple, set)) else []
            if not isinstance(years, int):
                 logging.warning(f"Expected 'years' to be an integer, got {type(years)}. Attempting conversion to int.")
                 try:
                     years = int(float(years)) # Allow float conversion then int
                 except (ValueError, TypeError):
                     logging.error("Could not convert 'years' to integer. Setting to 0.")
                     years = 0

            logging.info(f"Successfully extracted {len(skills)} skills and {years} relevant years of experience for role '{job_role}'")
            return skills, years
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini JSON response. Error: {str(e)}. Response attempted: {cleaned_response[:200]}")
            # Attempt basic regex fallback if JSON parsing fails
            skills_fallback = re.findall(r'"([^"]+)"', cleaned_response)
            years_match = re.search(r'"years":\s*(\d+)', cleaned_response)
            years_fallback = int(years_match.group(1)) if years_match else 0
            logging.warning(f"Using regex fallback: Found {len(skills_fallback)} potential skills and {years_fallback} years.")
            return skills_fallback, years_fallback
        except ValueError as ve:
            logging.error(f"Data validation error: {ve}. Response: {cleaned_response}")
            return [], 0 # Return empty/zero on validation error

    except Exception as e:
        # Catch potential API errors, configuration errors, etc.
        logging.error(f"Error during Gemini API call for CV data extraction: {str(e)}")
        # Consider more specific error handling based on google.generativeai exceptions if needed
        return [], 0 # Return empty list and 0 years on error

# Cell 7: MongoDB Query for Required Skills
def get_required_skills(job_collection, job_role):
    try:
        # Case-insensitive search for job role might be more robust
        job_doc = job_collection.find_one({"Job_Role": {"$regex": f"^{re.escape(job_role)}$", "$options": "i"}})
        if not job_doc:
            logging.error(f"Job role '{job_role}' not found in database (case-insensitive search).")
            raise ValueError(f"Job role '{job_role}' not found in database")

        required_skills_raw = job_doc.get("Required_Skills")
        if not required_skills_raw or not isinstance(required_skills_raw, str):
             logging.error(f"Required_Skills field missing or not a string for job role '{job_role}'.")
             raise ValueError(f"Required skills format error for job role '{job_role}'")

        # Clean up skills: split by comma, strip whitespace, remove empty strings
        required_skills = [skill.strip() for skill in required_skills_raw.split(",") if skill.strip()]
        if not required_skills:
            logging.warning(f"No required skills derived after cleaning for job role '{job_role}'.")
            return [] # Return empty list if no valid skills found

        logging.info(f"Required skills for {job_role}: {required_skills}")
        return required_skills
    except Exception as e:
        logging.error(f"Error retrieving required skills for '{job_role}': {str(e)}")
        # Re-raise to indicate failure in the calling function
        raise ValueError(f"Error retrieving required skills: {str(e)}")

# Cell 8: Experience Weight Assignment
def assign_experience_weight(years):
    """Assigns a weight based on years of relevant experience."""
    if not isinstance(years, (int, float)) or years < 0:
        logging.warning(f"Invalid years value '{years}' received. Assigning weight 0.")
        return 0
    years = int(years) # Ensure integer comparison

    if years > 15:
        logging.info(f"Assigning experience weight 3 for {years} years.")
        return 3  # High (Expert/Lead)
    elif 8 <= years <= 15:
        logging.info(f"Assigning experience weight 2 for {years} years.")
        return 2  # Medium (Senior)
    elif 1 <= years <= 7:
        logging.info(f"Assigning experience weight 1 for {years} years.")
        return 1  # Low (Junior/Mid)
    else: # 0 years or less than 1 year after rounding
        logging.info(f"Assigning experience weight 0 for {years} years.")
        return 0  # None

# Cell 9: OPTIMIZED - Combined function to analyze skill weights and skill gaps in a single API call
def analyze_job_requirements(job_role, required_skills, cv_skills):
    """
    Analyzes skill relevance (weights) and identifies missing skills using Gemini.
    """
    if not required_skills:
        logging.warning("No required skills provided for analysis. Returning empty results.")
        return {}, []
    if not cv_skills:
        logging.warning("No CV skills provided for analysis. All required skills will be marked as missing.")
        # Create default weights (e.g., 1) if needed, or handle this case downstream
        weights = {skill: 1 for skill in required_skills}
        return weights, required_skills # All required skills are missing

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        num_skills = len(required_skills)
        max_weight = max(5, num_skills) # Ensure a reasonable max weight even for few skills

        # Use JSON dumps for reliable formatting of skill lists in the prompt
        required_skills_json = json.dumps(required_skills)
        cv_skills_json = json.dumps(cv_skills)

        prompt = (
            f"You are an expert technical recruiter specializing in the '{job_role}' domain.\n\n"
            f"TASK 1 - SKILL WEIGHTING:\n"
            f"Analyze the following list of skills required for the '{job_role}' position:\n{required_skills_json}\n"
            f"Assign an integer weight to EACH required skill based on its importance and relevance specifically for the '{job_role}'. "
            f"Use a scale from 1 (least important, e.g., general soft skill if listed) to {max_weight} (most critical technical skill for this role). "
            f"Ensure core technical skills directly related to '{job_role}' receive higher weights. Try to distribute weights reasonably, but uniqueness is not strictly required if skills have similar importance.\n\n"
            f"TASK 2 - SKILL GAP ANALYSIS:\n"
            f"Compare the candidate's skills extracted from their CV:\n{cv_skills_json}\n"
            f"against the required skills list:\n{required_skills_json}\n"
            f"Identify ONLY the required skills that are ABSENT from the candidate's CV skills. "
            f"Consider semantic similarity and common variations (e.g., 'Machine Learning' matches 'ML', 'AWS' matches 'Amazon Web Services', 'Python Programming' matches 'Python', 'SQL' matches 'Database Management' ONLY IF context implies database querying). Be conservative with semantic matching; if unsure, list the skill as missing.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return a single, valid JSON object containing exactly two keys:\n"
            f"- 'weights': An object where keys are the required skill strings and values are their assigned integer weights.\n"
            f"- 'missing_skills': An array of strings, listing only the required skills identified as missing from the CV skills.\n\n"
            f"Example JSON Output:\n"
            f"{{\"weights\": {{\"Python\": {max_weight}, \"Machine Learning\": {max_weight-1}, \"SQL\": {max_weight // 2}, \"Communication\": 1}}, \"missing_skills\": [\"SQL\"]}}\n\n"
            f"IMPORTANT: Ensure the output is ONLY the JSON object, with no surrounding text, comments, or markdown formatting like ```json ... ```."
        )

        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw job analysis response (first 100 chars): {raw_response[:100]}...")

        # Clean response
        cleaned_response = re.sub(r"^```json\s*", "", raw_response)
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            analysis_data = json.loads(cleaned_response)
            if not isinstance(analysis_data, dict):
                raise ValueError("Parsed JSON is not a dictionary.")
            if "weights" not in analysis_data or "missing_skills" not in analysis_data:
                 logging.error(f"Gemini response missing required keys ('weights', 'missing_skills'). Response: {cleaned_response}")
                 # Fallback: Assign weight 1 to all, assume all are missing initially if structure is wrong
                 return {skill: 1 for skill in required_skills}, required_skills

            weights = analysis_data.get("weights", {})
            missing_skills = analysis_data.get("missing_skills", [])

            # --- Validation ---
            if not isinstance(weights, dict):
                 logging.warning(f"Expected 'weights' to be a dict, got {type(weights)}. Using fallback.")
                 weights = {skill: 1 for skill in required_skills} # Fallback weights

            if not isinstance(missing_skills, list):
                logging.warning(f"Expected 'missing_skills' to be a list, got {type(missing_skills)}. Using fallback.")
                # Attempt recovery or return all as missing
                missing_skills = list(missing_skills) if isinstance(missing_skills, (tuple, set)) else required_skills

            # Ensure all required skills have a weight (assign default if missing)
            for skill in required_skills:
                if skill not in weights:
                    logging.warning(f"Skill '{skill}' missing from weights response. Assigning default weight 1.")
                    weights[skill] = 1
                elif not isinstance(weights[skill], int):
                     logging.warning(f"Weight for skill '{skill}' is not an integer ({type(weights[skill])}). Setting to 1.")
                     weights[skill] = 1

            # Ensure missing skills are actually from the required list
            validated_missing = [ms for ms in missing_skills if ms in required_skills]
            if len(validated_missing) != len(missing_skills):
                logging.warning("Gemini reported missing skills not in the original required list. Filtering them out.")
                missing_skills = validated_missing
            # --- End Validation ---

            logging.info(f"Analyzed weights for {len(weights)} skills and identified {len(missing_skills)} missing skills for role '{job_role}'")
            return weights, missing_skills
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini JSON response for job analysis. Error: {str(e)}. Response: {cleaned_response[:200]}")
            # Fallback: assign default weights, assume all missing
            return {skill: 1 for skill in required_skills}, required_skills
        except ValueError as ve:
            logging.error(f"Data validation error in job analysis: {ve}. Response: {cleaned_response}")
            return {skill: 1 for skill in required_skills}, required_skills # Fallback

    except Exception as e:
        logging.error(f"Error during Gemini API call for job analysis: {str(e)}")
        # Fallback: assign default weights, assume all missing
        return {skill: 1 for skill in required_skills}, required_skills

# Cell 10: Updated CV Ranking Function using optimized API calls
def rank_cvs(cv_folder, job_role):
    """
    Ranks CVs in a folder based on skills match and relevant experience for a specific job role.
    """
    if not os.path.isdir(cv_folder):
        logging.error(f"CV folder does not exist or is not a directory: {cv_folder}")
        raise ValueError(f"Invalid CV folder path: {cv_folder}")

    try:
        job_collection, _ = connect_to_mongodb()
        required_skills = get_required_skills(job_collection, job_role)
        if not required_skills:
             logging.error(f"No required skills found for job role '{job_role}'. Cannot perform ranking.")
             # Return empty lists and dict if no required skills are defined
             return [], {}

        cv_results = []
        cv_texts = {}  # Store CV text for chatbot

        logging.info(f"Starting CV processing in folder: {cv_folder} for job role: {job_role}")
        processed_files = 0
        error_files = 0

        for cv_file in os.listdir(cv_folder):
            # Process only PDF files
            if cv_file.lower().endswith('.pdf'):
                cv_path = os.path.join(cv_folder, cv_file)
                logging.info(f"--- Processing CV file: {cv_file} ---")
                try:
                    # Extract text (handles potential errors inside)
                    cv_text = extract_cv_text(cv_path)
                    cv_texts[cv_file] = cv_text  # Store CV text regardless of subsequent errors for potential debugging

                    # Optional: Extract skills section (currently not used by Gemini call but kept for potential future use)
                    # skills_section_text = extract_skills_section(cv_text)

                    # Call Gemini to extract skills and *relevant* experience years
                    # Pass the full CV text and the specific job role
                    cv_skills, years = extract_cv_data_with_gemini(cv_text, job_role) # MODIFIED CALL

                    if not cv_skills and years == 0:
                        logging.warning(f"No skills or experience extracted from {cv_file} by Gemini. Skipping ranking for this CV.")
                        error_files += 1
                        continue # Skip if Gemini returned nothing useful

                    # Assign experience weight based on relevant years
                    exp_weight = assign_experience_weight(years)

                    # Call Gemini to get skill weights and missing skills
                    skill_weights, missing_skills = analyze_job_requirements(job_role, required_skills, cv_skills)

                    # Calculate skill score based on weights of matched skills
                    matched_skills = [skill for skill in cv_skills if skill in skill_weights] # Use skill_weights keys as the definitive list
                    skill_score = sum(skill_weights.get(skill, 0) for skill in matched_skills)

                    # Calculate skill match percentage
                    num_required = len(required_skills)
                    num_matched = len(set(required_skills) & set(cv_skills)) # Count unique matches
                    skill_match_percent = (num_matched / num_required * 100) if num_required > 0 else 0

                    # Calculate total score (can be adjusted based on relative importance)
                    # Example: Simple sum. Could be weighted: total_score = (exp_weight * 0.3) + (skill_score * 0.7)
                    total_score = exp_weight + skill_score

                    cv_results.append({
                        "cv_file": cv_file,
                        "years_experience": years,          # Relevant years
                        "experience_weight": exp_weight,
                        "cv_skills": cv_skills,             # Skills found in CV
                        "required_skills": required_skills, # Skills needed for the job
                        "matched_skills": list(set(required_skills) & set(cv_skills)), # Explicit list of matches
                        "missing_skills": missing_skills,   # Required skills not found
                        "skill_score": skill_score,         # Weighted score of matched skills
                        "skill_match_percent": round(skill_match_percent, 1), # Percentage of required skills matched
                        "total_score": total_score          # Combined score
                    })
                    logging.info(f"Successfully processed {cv_file}: Score = {total_score}, Relevant Years = {years}, Match = {round(skill_match_percent, 1)}%")
                    processed_files += 1
                except ValueError as ve: # Catch specific errors from text extraction etc.
                    logging.error(f"Skipping {cv_file} due to processing error: {str(ve)}")
                    error_files += 1
                    continue # Skip to next file
                except Exception as e:
                    logging.error(f"Unexpected error processing {cv_file}: {str(e)}", exc_info=True) # Log stack trace for unexpected errors
                    error_files += 1
                    continue # Skip to next file
            else:
                logging.info(f"Skipping non-PDF file: {cv_file}")


        # Sort CVs by total score in descending order
        ranked_cvs = sorted(cv_results, key=lambda x: x["total_score"], reverse=True)

        logging.info(f"--- CV Ranking Complete for role '{job_role}' ---")
        logging.info(f"Successfully processed: {processed_files} CVs")
        logging.info(f"Skipped due to errors: {error_files} CVs")
        logging.info(f"Total CVs ranked: {len(ranked_cvs)}")
        if not ranked_cvs:
            logging.warning("No CVs were successfully ranked.")

        return ranked_cvs, cv_texts
    except ValueError as ve: # Catch errors from connect_to_mongodb or get_required_skills
        logging.error(f"Setup error during CV ranking: {str(ve)}")
        return [], {} # Return empty if setup fails
    except Exception as e:
        logging.error(f"Critical error during CV ranking process: {str(e)}", exc_info=True)
        return [], {} # Return empty on unexpected critical failure

# Cell 11: Tkinter GUI with Chatbot
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox # Added messagebox

# Handler to redirect logging to the Tkinter text widget
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        # Ensure GUI updates are thread-safe if using threading later
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END) # Auto-scroll

def create_gui():
    root = tk.Tk()
    root.title("CV Ranking System with Chatbot (Gemini)")
    root.geometry("1100x750") # Slightly larger window

    # --- Global variables for GUI state ---
    cv_texts = {}       # Dictionary to store CV text: {filename: text}
    ranked_cvs = []     # List to store analysis results: [ {cv_data_dict}, ... ]
    job_roles = []      # List of available job roles
    llm = None          # Placeholder for the LangChain LLM

    # --- Initialize LangChain Gemini model ---
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Or another suitable model
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2, # Lower temperature for more factual answers
            # Add safety_settings if needed: safety_settings=...
            convert_system_message_to_human=True # Often helps with compatibility
        )
        logging.info("LangChain Gemini LLM initialized successfully.")
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize LangChain Gemini LLM: {str(e)}")
        messagebox.showerror("Initialization Error", f"Failed to initialize Gemini LLM. Chatbot will not function.\nError: {str(e)}")
        llm = None # Ensure llm is None if initialization fails

    # --- Chatbot Interaction Logic ---
    def ask_cv_question(cv_filename, question):
        nonlocal llm, ranked_cvs, cv_texts # Access shared state
        if not llm:
            return "Chatbot is not available due to an initialization error."
        if not cv_filename or cv_filename == "Select CV after analysis" or cv_filename not in cv_texts:
            return "Please select a valid CV from the dropdown first."
        if not question:
            return "Please enter a question."

        try:
            job_role = job_role_var.get()
            cv_text = cv_texts.get(cv_filename, "CV text not found.")
            # Find the analysis data for the selected CV
            selected_cv_analysis = next((item for item in ranked_cvs if item["cv_file"] == cv_filename), None)
            analysis_summary = json.dumps(selected_cv_analysis, indent=2) if selected_cv_analysis else "Analysis data not found for this CV."

            # --- Relevance Check (Optional but Recommended) ---
            # relevance_prompt = [
            #     SystemMessage(content="Is the following question related to analyzing a CV, comparing CVs, or discussing job skills/experience? Answer ONLY 'yes' or 'no'."),
            #     HumanMessage(content=f"Question: {question}")
            # ]
            # relevance_response = llm.invoke(relevance_prompt).content.strip().lower()
            # logging.info(f"Chatbot relevance check for '{question}': {relevance_response}")
            # if 'no' in relevance_response:
            #     return "I can only answer questions about the CVs, job roles, skills, experience, and the ranking analysis."
            # --- End Relevance Check ---

            # --- Answering Prompt ---
            # Provide context: job role, specific CV text, and its analysis results
            system_prompt = (
                "You are a helpful AI assistant expert in recruitment and CV analysis. "
                "Answer the user's question based *only* on the provided CV text and its analysis results. "
                "Be concise and factual. If the information isn't in the provided text or analysis, state that explicitly. "
                "Do not make assumptions or provide information beyond the context given."
            )
            human_prompt_content = (
                f"Context:\n"
                f"Job Role Analyzed For: {job_role}\n"
                f"Selected CV Filename: {cv_filename}\n\n"
                f"CV Analysis Summary:\n{analysis_summary}\n\n"
                f"Full CV Text:\n```text\n{cv_text}\n```\n\n"
                f"User Question: {question}"
            )
            answer_prompt = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt_content)
            ]

            logging.info(f"Sending question to LLM for CV '{cv_filename}': {question}")
            response = llm.invoke(answer_prompt)
            answer = response.content.strip()
            logging.info(f"LLM response: {answer[:100]}...")
            return answer

        except Exception as e:
            logging.error(f"Error processing chatbot query: {str(e)}", exc_info=True)
            return "Sorry, an error occurred while processing your question. Please check the logs."

    # --- GUI Setup ---
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Row 0: Folder Selection
    ttk.Label(main_frame, text="Select CV Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    folder_entry = ttk.Entry(main_frame, width=60)
    folder_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    def browse_folder():
        folder_path = filedialog.askdirectory(title="Select Folder Containing CV PDFs")
        if folder_path:
            folder_entry.delete(0, tk.END)
            folder_entry.insert(0, folder_path)
            logging.info(f"CV folder selected: {folder_path}")
    browse_button = ttk.Button(main_frame, text="Browse...", command=browse_folder)
    browse_button.grid(row=0, column=2, padx=5, pady=5)

    # Row 1: Job Role Selection
    ttk.Label(main_frame, text="Select Job Role:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    job_role_var = tk.StringVar()
    try:
        # Fetch job roles at startup
        _, job_roles = connect_to_mongodb()
        if not job_roles:
            job_roles = ["No roles found"] # Placeholder if DB is empty
            messagebox.showwarning("Database Info", "No job roles found in the database. Please check the MongoDB collection 'jobrole_skill'.")
    except Exception as e:
        job_roles = ["Error loading roles"]
        messagebox.showerror("Database Error", f"Failed to fetch job roles from MongoDB: {str(e)}")
        logging.error(f"Failed to fetch job roles: {str(e)}")

    job_role_dropdown = ttk.Combobox(main_frame, textvariable=job_role_var, values=job_roles, state="readonly", width=57)
    if job_roles:
        job_role_var.set(job_roles[0])
    job_role_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    # Row 2: Run Analysis Button
    def run_analysis_thread():
         # Disable button during analysis
        run_button.config(state=tk.DISABLED)
        browse_button.config(state=tk.DISABLED)
        job_role_dropdown.config(state=tk.DISABLED)
        root.update_idletasks() # Ensure GUI updates

        nonlocal cv_texts, ranked_cvs # Modify global state
        cv_folder = folder_entry.get()
        job_role = job_role_var.get()

        # Clear previous results and log
        result_text.config(state='normal')
        result_text.delete(1.0, tk.END)
        result_text.config(state='disabled')
        chat_text.config(state='normal')
        chat_text.delete(1.0, tk.END)
        chat_text.config(state='disabled')

        if not cv_folder or not os.path.isdir(cv_folder):
            logging.error("Invalid CV folder selected.")
            messagebox.showerror("Error", "Please select a valid folder containing CVs.")
            run_button.config(state=tk.NORMAL) # Re-enable button
            browse_button.config(state=tk.NORMAL)
            job_role_dropdown.config(state=tk.NORMAL if job_roles != ["Error loading roles"] else tk.DISABLED)
            return
        if not job_role or job_role in ["No roles found", "Error loading roles"]:
            logging.error("Invalid job role selected.")
            messagebox.showerror("Error", "Please select a valid job role.")
            run_button.config(state=tk.NORMAL)
            browse_button.config(state=tk.NORMAL)
            job_role_dropdown.config(state=tk.NORMAL if job_roles != ["Error loading roles"] else tk.DISABLED)
            return

        try:
            logging.info(f"--- Starting Analysis for Job Role: {job_role} ---")
            # Run the core ranking function
            ranked_cvs, cv_texts = rank_cvs(cv_folder, job_role)

            # --- Update Results Text Area ---
            result_text.config(state='normal')
            result_text.insert(tk.END, f"--- Analysis Results for Job Role: {job_role} ---\n\n")
            if ranked_cvs:
                 result_text.insert(tk.END, f"Ranked {len(ranked_cvs)} CVs:\n")
                 for i, cv in enumerate(ranked_cvs, 1):
                    result_text.insert(tk.END, f"\nRank {i}: {cv['cv_file']}\n")
                    result_text.insert(tk.END, f"  Total Score: {cv['total_score']}\n")
                    result_text.insert(tk.END, f"  Relevant Experience: {cv['years_experience']} years (Weight: {cv['experience_weight']})\n")
                    result_text.insert(tk.END, f"  Skill Match: {cv['skill_match_percent']}% (Score: {cv['skill_score']})\n")
                    result_text.insert(tk.END, f"  Matched Skills ({len(cv['matched_skills'])}): {', '.join(cv['matched_skills'])}\n")
                    result_text.insert(tk.END, f"  Missing Skills ({len(cv['missing_skills'])}): {', '.join(cv['missing_skills'])}\n")
                    # Optionally show all CV skills:
                    # result_text.insert(tk.END, f"  CV Skills ({len(cv['cv_skills'])}): {', '.join(cv['cv_skills'])}\n")
            else:
                result_text.insert(tk.END, "No CVs were successfully processed or ranked. Please check logs and CV files.\n")
            result_text.config(state='disabled')
            result_text.yview(tk.END) # Scroll to end

            # --- Update CV Dropdown for Chatbot ---
            cv_files = [cv['cv_file'] for cv in ranked_cvs] if ranked_cvs else ["No CVs ranked"]
            cv_dropdown['values'] = cv_files
            if cv_files[0] != "No CVs ranked":
                cv_var.set(cv_files[0])
            else:
                 cv_var.set(cv_files[0])
                 cv_dropdown.config(state=tk.DISABLED)
            if cv_files[0] != "No CVs ranked":
                cv_dropdown.config(state="readonly")


        except Exception as e:
            logging.error(f"Analysis failed critically: {str(e)}", exc_info=True)
            messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis:\n{str(e)}")
        finally:
             # Re-enable button after analysis finishes or fails
            run_button.config(state=tk.NORMAL)
            browse_button.config(state=tk.NORMAL)
            job_role_dropdown.config(state=tk.NORMAL if job_roles != ["Error loading roles"] else tk.DISABLED)

    # Run analysis in a separate thread to avoid freezing the GUI (Recommended for longer tasks)
    import threading
    def start_analysis():
        # Create and start the thread
        analysis_thread = threading.Thread(target=run_analysis_thread, daemon=True)
        analysis_thread.start()

    run_button = ttk.Button(main_frame, text="Run Analysis", command=start_analysis)
    run_button.grid(row=1, column=2, padx=5, pady=5) # Moved next to job role

    # Row 3: Results Area (Log and Ranking Output)
    results_frame = ttk.LabelFrame(main_frame, text="Analysis Log & Results", padding="5")
    results_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
    results_frame.columnconfigure(0, weight=1)
    results_frame.rowconfigure(0, weight=1)

    result_text = scrolledtext.ScrolledText(results_frame, width=120, height=18, state='disabled', wrap=tk.WORD) # Increased height
    result_text.grid(row=0, column=0, sticky="nsew")

    # Setup logging handler
    text_handler = TextHandler(result_text)
    text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # Keep existing handlers if any (like console output), add the GUI handler
    logging.getLogger().addHandler(text_handler)
    logging.getLogger().setLevel(logging.INFO) # Ensure INFO level messages are captured


    # Row 4: Chatbot Area
    chatbot_frame = ttk.LabelFrame(main_frame, text="Chatbot - Ask about a specific CV", padding="5")
    chatbot_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
    chatbot_frame.columnconfigure(0, weight=1)
    # Configure row weights for chatbot elements
    chatbot_frame.rowconfigure(1, weight=1) # Chat history area expands


    # Row 4, Col 0: CV Selection for Chatbot
    ttk.Label(chatbot_frame, text="Select CV:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    cv_var = tk.StringVar()
    cv_dropdown = ttk.Combobox(chatbot_frame, textvariable=cv_var, values=["Select CV after analysis"], state=tk.DISABLED, width=40)
    cv_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")


    # Row 4, Col 1: Chat History
    chat_text = scrolledtext.ScrolledText(chatbot_frame, width=100, height=10, state='disabled', wrap=tk.WORD) # Increased height
    chat_text.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")


    # Row 4, Col 2: Question Entry & Send Button
    question_entry = ttk.Entry(chatbot_frame, width=80)
    question_entry.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    question_entry.bind("<Return>", lambda event: send_question()) # Bind Enter key


    def send_question():
        nonlocal llm # Access the LLM instance
        question = question_entry.get().strip()
        selected_cv = cv_var.get()

        if not llm:
             messagebox.showerror("Chatbot Error", "Chatbot is not initialized. Cannot send question.")
             return

        if selected_cv == "Select CV after analysis" or selected_cv == "No CVs ranked":
             messagebox.showwarning("Chatbot", "Please run the analysis and select a valid CV from the dropdown.")
             return
        if not question:
            messagebox.showwarning("Chatbot", "Please enter a question.")
            return

        # Add question to chat history immediately
        chat_text.config(state='normal')
        chat_text.insert(tk.END, f"You: {question}\n")
        chat_text.config(state='disabled')
        chat_text.yview(tk.END)
        question_entry.delete(0, tk.END)
        root.update_idletasks() # Show question before waiting for response

        # Get response (can run in thread if LLM call is slow)
        try:
             # Consider running ask_cv_question in a thread for responsiveness
             response = ask_cv_question(selected_cv, question)
             chat_text.config(state='normal')
             chat_text.insert(tk.END, f"Chatbot: {response}\n\n") # Add extra newline
             chat_text.config(state='disabled')
             chat_text.yview(tk.END)
        except Exception as e:
             # Handle potential errors during the ask_cv_question call itself
             logging.error(f"Error getting chatbot response: {e}", exc_info=True)
             chat_text.config(state='normal')
             chat_text.insert(tk.END, f"Chatbot Error: Could not get response. Check logs.\n\n")
             chat_text.config(state='disabled')
             chat_text.yview(tk.END)


    send_button = ttk.Button(chatbot_frame, text="Send", command=send_question)
    send_button.grid(row=2, column=2, padx=5, pady=5)


    # Configure main_frame resizing
    main_frame.columnconfigure(1, weight=1) # Allow entry/combobox to expand
    main_frame.rowconfigure(3, weight=1)    # Allow results area to expand
    main_frame.rowconfigure(4, weight=1)    # Allow chatbot area to expand


    # Start the Tkinter event loop
    logging.info("GUI Created. Ready for input.")
    root.mainloop()

# Cell 12: Run the application
if __name__ == "__main__":
    # --- Basic Pre-checks ---
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE": # Basic check
        print("ERROR: GOOGLE_API_KEY is not set correctly in the script.")
        logging.error("GOOGLE_API_KEY is not set correctly.")
        # Optionally exit or show a GUI error if Tkinter is already imported
        # exit(1) # Or handle more gracefully
    if not MONGODB_URI or "YOUR_MONGODB_URI" in MONGODB_URI: # Basic check
        print("ERROR: MONGODB_URI is not set correctly in the script.")
        logging.error("MONGODB_URI is not set correctly.")
        # exit(1) # Or handle more gracefully

    # Attempt to check Tesseract path existence if set explicitly (optional)
    # try:
    #     if pytesseract.pytesseract.tesseract_cmd and not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
    #         logging.warning(f"Tesseract path set to '{pytesseract.pytesseract.tesseract_cmd}' but file not found. OCR may fail.")
    # except AttributeError:
    #      logging.info("Tesseract path not explicitly set in the script. Assuming it's in system PATH.")


    # --- Start the GUI ---
    create_gui()

