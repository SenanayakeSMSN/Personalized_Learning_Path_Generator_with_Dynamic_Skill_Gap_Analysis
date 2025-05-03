"""
THIS THE STREAMLIT CODE FOR TESTQ_2

"""

# Cell 2: Import Libraries and Setup
import os
import json
import logging
import re
from datetime import datetime
from pymongo import MongoClient
from PIL import Image
import io # Needed for Streamlit logging

# --- Streamlit Import ---
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Security Warning ---
# Hardcoding keys as requested for now.
# REMEMBER TO MOVE THESE TO ENVIRONMENT VARIABLES OR SECRETS LATER.
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo" # Hardcoded Google API Key
MONGODB_URI = "mongodb+srv://akilapremarathna0:123@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority" # Hardcoded MongoDB URI
# --- End Security Warning ---

# Check specifically for generic placeholders (not the hardcoded values above)
_MISSING_CONFIG = []
if GOOGLE_API_KEY == "YOUR_API_KEY_HERE": # Only check for generic placeholder
    _MISSING_CONFIG.append("Google API Key")
if MONGODB_URI == "YOUR_MONGODB_URI": # Only check for generic placeholder
    _MISSING_CONFIG.append("MongoDB URI")

# Check if the specific default/hardcoded keys are being used (for warning)
_USING_DEFAULTS = []
# Add check for the specific keys being used (useful if script is shared)
if GOOGLE_API_KEY == "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo":
     _USING_DEFAULTS.append("Google API Key")
if MONGODB_URI == "mongodb+srv://akilapremarathna0:123@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority":
     _USING_DEFAULTS.append("MongoDB URI")


# Import dependencies
try:
    import pdfplumber
except ImportError:
    st.error("pdfplumber is not installed. Run: pip install pdfplumber")
    st.stop()

try:
    import google.generativeai as genai
    # Configure GenAI Key here IF using the non-Langchain functions directly
    # Check if key is NOT a generic placeholder before configuring
    if GOOGLE_API_KEY not in ["YOUR_API_KEY_HERE"]:
         try:
             genai.configure(api_key=GOOGLE_API_KEY)
         except Exception as genai_err:
             st.error(f"Failed to configure Google Generative AI: {genai_err}. Check if the API key is valid.")
             # Don't stop necessarily, maybe only parts of the app fail
    else:
        # Error handled later if key is missing and needed
        pass
except ImportError:
    st.error("google-generativeai is not installed. Run: pip install google-generativeai")
    st.stop()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    # LangChain LLM Initialization moved to main app logic where needed
except ImportError:
    st.error("langchain-google-genai is not installed. Run: pip install langchain-google-genai")
    st.stop()

try:
    import pytesseract
    # Explicitly set Tesseract path (adjust if installed elsewhere)
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Example
except ImportError:
    st.warning("pytesseract or PIL is not installed (pip install pytesseract Pillow). OCR fallback for PDFs will not work.")
except Exception as e:
    st.warning(f"Could not configure pytesseract path: {e}. Ensure Tesseract is installed and in your PATH or set the path manually. OCR fallback may fail.")


from langchain_core.messages import HumanMessage, SystemMessage

# Cell 3: MongoDB Connection
@st.cache_resource(ttl=3600) # Cache resource for 1 hour
def get_mongo_client():
    """Connects to MongoDB Client."""
    # Check for generic placeholder before connecting
    if MONGODB_URI in ["YOUR_MONGODB_URI"]:
         logging.error("MONGODB_URI is not set (using generic placeholder). Please configure it.")
         st.error("MongoDB URI is not configured correctly (using generic placeholder). Cannot connect to database.")
         st.stop()
    try:
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        st.error(f"Failed to connect to MongoDB: {str(e)}. Check the URI and network access.")
        st.stop()

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_job_roles_from_db():
    """Fetches the list of job roles from MongoDB."""
    client = get_mongo_client()
    # Client connection success/failure handled within get_mongo_client
    try:
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
        collection.create_index("Job_Role")
        job_roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1}) if "Job_Role" in doc and doc["Job_Role"]]
        job_roles = sorted(list(set(job_roles)))
        if not job_roles:
            logging.warning("No job roles found in the database collection 'jobrole_skill'.")
            return ["No job roles found in DB"]
        logging.info(f"Fetched {len(job_roles)} unique job roles from DB.")
        return ["Select Job Role"] + job_roles
    except Exception as e:
        logging.error(f"Failed to fetch job roles: {str(e)}")
        st.error(f"Failed to fetch job roles from MongoDB: {str(e)}")
        return ["Error loading roles"]

def get_job_collection():
    """Gets the specific MongoDB collection object."""
    client = get_mongo_client()
    # Client connection success/failure handled within get_mongo_client
    try:
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
        return collection
    except Exception as e:
        logging.error(f"Failed to get job collection: {str(e)}")
        st.error(f"Failed to access MongoDB collection: {str(e)}")
        return None

# Cell 4: Enhanced CV Text Extraction (Unchanged)
def extract_cv_text(cv_path):
    """Extracts text from a PDF CV, attempting OCR as a fallback."""
    text = ""
    try:
        with pdfplumber.open(cv_path) as pdf:
            logging.info(f"Processing PDF: {os.path.basename(cv_path)}")
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n\n"
                    logging.debug(f"Extracted text from page {page_number} using pdfplumber") # Debug level
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
                    except ImportError:
                        logging.error(f"OCR attempted on page {page_number} but pytesseract/PIL not installed correctly.")
                        text += "\n\n"
                    except Exception as ocr_e:
                        # Check if it's a TesseractNotFoundError
                        if "Tesseract is not installed or" in str(ocr_e):
                             logging.error(f"OCR failed for page {page_number}: Tesseract not found. Please install Tesseract and ensure it's in your PATH or configure pytesseract.pytesseract.tesseract_cmd.")
                             st.warning(f"OCR failed for page {page_number} in {os.path.basename(cv_path)}: Tesseract not found. Results may be incomplete if PDF requires OCR.")
                        else:
                            logging.error(f"OCR failed for page {page_number}: {str(ocr_e)}. Check Tesseract installation/path.")
                            st.warning(f"OCR failed for page {page_number} in {os.path.basename(cv_path)}: {str(ocr_e)}. Results may be incomplete.")
                        text += "\n\n" # Add space to maintain structure
            if not text.strip():
                logging.error(f"No text extracted from CV '{os.path.basename(cv_path)}' after attempting both standard extraction and OCR.")
                st.warning(f"No text could be extracted from the CV PDF '{os.path.basename(cv_path)}'. Check file content/format or Tesseract setup. This CV will likely have a score of 0.")
                return ""
            logging.info(f"Successfully extracted text from CV '{os.path.basename(cv_path)}'. Length: {len(text)} chars.")
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error(f"PDF file '{os.path.basename(cv_path)}' is password-protected. Cannot process.")
        st.warning(f"'{os.path.basename(cv_path)}' is password-protected and cannot be processed.")
        return None
    except FileNotFoundError:
        logging.error(f"CV file not found at path: {cv_path}")
        st.error(f"Internal Error: CV file not found at path: {cv_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading CV '{os.path.basename(cv_path)}': {str(e)}")
        st.warning(f"Error processing CV '{os.path.basename(cv_path)}': {str(e)}. It will be skipped.")
        return None

# Cell 5: Skills Section Extraction (Optional Enhancement - Not used by Gemini in this version) (Unchanged)
def extract_skills_section(cv_text):
    """Attempts to extract a dedicated skills section using regex."""
    try:
        match = re.search(
            r"(?i)\b(skills|technical\s+skills|key\s+skills|core\s+competencies|technical\s+expertise|proficiencies)\b[:\s]*\n?(.*?)(?=\n\s*\b(experience|work\s+history|employment|education|projects|publications|awards|certifications|references|contact|personal\s+details)\b|\Z)",
            cv_text, re.DOTALL
        )
        if match:
            skills_text = match.group(2).strip()
            skills_text = re.sub(r'^\s*[-*•]\s*', '', skills_text, flags=re.MULTILINE)
            skills_text = re.sub(r'\s+', ' ', skills_text)
            if skills_text:
                 logging.info(f"Extracted potential skills section: {skills_text[:100]}...")
                 return skills_text
            else:
                 logging.warning("Found skills section header but no content extracted.")
                 return cv_text
        logging.warning("No explicit skills section found using regex patterns. Using full CV text for analysis.")
        return cv_text
    except Exception as e:
        logging.error(f"Error during regex extraction of skills section: {str(e)}")
        return cv_text

# Cell 6: MODIFIED - Combined function to extract both skills and *role-specific* experience (Unchanged Logic, relies on GOOGLE_API_KEY)
def extract_cv_data_with_gemini(cv_text, job_role):
    """
    Extracts skills and calculates relevant years of experience using Gemini,
    tailored to a specific job role.
    """
    # Check for generic placeholder API key
    if GOOGLE_API_KEY in ["YOUR_API_KEY_HERE"]:
         logging.error("GOOGLE_API_KEY is not set (using generic placeholder). Cannot call Gemini.")
         st.error("Google API Key is not configured correctly (using generic placeholder). Cannot extract CV data using Gemini.")
         return [], 0
    if not cv_text or not cv_text.strip():
        logging.error("CV text is empty, cannot extract data.")
        return [], 0
    if not job_role or not job_role.strip():
        logging.error("Job role is empty, cannot calculate relevant experience.")
        return [], 0

    try:
        # GenAI should be configured at the start
        model = genai.GenerativeModel("gemini-1.5-flash")

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

        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw response for CV data (first 100 chars): {raw_response[:100]}...")

        cleaned_response = re.sub(r"^```json\s*", "", raw_response)
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            cv_data = json.loads(cleaned_response)
            if not isinstance(cv_data, dict):
                 raise ValueError("Parsed JSON is not a dictionary.")
            if "skills" not in cv_data or "years" not in cv_data:
                logging.error(f"Gemini response missing required keys ('skills', 'years'). Response: {cleaned_response}")
                skills_fallback = re.findall(r'"([^"]+)"', cleaned_response)
                years_match = re.search(r'"years":\s*(\d+)', cleaned_response)
                years_fallback = int(years_match.group(1)) if years_match else 0
                st.warning(f"Gemini response for skills/years was malformed. Using fallback extraction (found {len(skills_fallback)} skills, {years_fallback} years). Results may be inaccurate.")
                return skills_fallback, years_fallback

            skills = cv_data.get("skills", [])
            years = cv_data.get("years", 0)

            if not isinstance(skills, list):
                logging.warning(f"Expected 'skills' to be a list, got {type(skills)}. Attempting conversion or returning empty.")
                skills = list(skills) if isinstance(skills, (tuple, set)) else []
            skills = [str(s) for s in skills if isinstance(s, (str, int, float))]

            if not isinstance(years, int):
                 logging.warning(f"Expected 'years' to be an integer, got {type(years)}. Attempting conversion to int.")
                 try:
                     years = int(float(years))
                 except (ValueError, TypeError):
                     logging.error("Could not convert 'years' to integer. Setting to 0.")
                     years = 0

            logging.info(f"Successfully extracted {len(skills)} skills and {years} relevant years of experience for role '{job_role}'")
            return skills, years
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini JSON response for skills/years. Error: {str(e)}. Response attempted: {cleaned_response[:200]}")
            skills_fallback = re.findall(r'"([^"]+)"', cleaned_response)
            years_match = re.search(r'"years":\s*(\d+)', cleaned_response)
            years_fallback = int(years_match.group(1)) if years_match else 0
            logging.warning(f"Failed to parse Gemini JSON. Using regex fallback: Found {len(skills_fallback)} potential skills and {years_fallback} years.")
            st.warning(f"Gemini response for skills/years was not valid JSON. Using fallback extraction (found {len(skills_fallback)} skills, {years_fallback} years). Results may be inaccurate.")
            return skills_fallback, years_fallback
        except ValueError as ve:
            logging.error(f"Data validation error after parsing skills/years: {ve}. Response: {cleaned_response}")
            st.warning(f"Data validation error processing Gemini response for skills/years: {ve}. Using default values (0 skills, 0 years).")
            return [], 0

    except Exception as e:
        logging.error(f"Error during Gemini API call for CV data extraction: {str(e)}")
        st.error(f"Error calling Gemini API for CV data extraction: {str(e)}. Check API key and network.")
        return [], 0

# Cell 7: MongoDB Query for Required Skills (Unchanged Logic, uses get_job_collection)
def get_required_skills(job_collection, job_role):
    """Fetches and cleans the list of required skills for a given job role."""
    if job_collection is None:
        raise ValueError("Database collection not available.")
    try:
        job_doc = job_collection.find_one({"Job_Role": {"$regex": f"^{re.escape(job_role)}$", "$options": "i"}})
        if not job_doc:
            logging.error(f"Job role '{job_role}' not found in database (case-insensitive search).")
            st.warning(f"Job role '{job_role}' not found in the database, or the document has no 'Required_Skills' field. Assuming no required skills for analysis.")
            return []

        required_skills_raw = job_doc.get("Required_Skills")
        if not required_skills_raw or not isinstance(required_skills_raw, str):
             logging.warning(f"Required_Skills field missing or not a string for job role '{job_role}'. No required skills loaded.")
             st.warning(f"'Required_Skills' field is missing or not a string for job role '{job_role}'. Assuming no required skills.")
             return []

        required_skills = [skill.strip() for skill in required_skills_raw.split(",") if skill.strip()]
        if not required_skills:
            logging.warning(f"No required skills derived after cleaning for job role '{job_role}'. Raw value was: '{required_skills_raw}'")
            st.warning(f"Required skills string for '{job_role}' was present but empty after cleaning. Assuming no required skills.")
            return []

        logging.info(f"Required skills for {job_role}: {required_skills}")
        return required_skills
    except Exception as e:
        logging.error(f"Error retrieving required skills for '{job_role}': {str(e)}")
        st.error(f"Error retrieving required skills for '{job_role}' from database: {str(e)}")
        raise ValueError(f"Error retrieving required skills for '{job_role}': {str(e)}")

# Cell 8: Experience Weight Assignment (Unchanged)
def assign_experience_weight(years):
    """Assigns a weight based on years of relevant experience."""
    if not isinstance(years, (int, float)) or years < 0:
        logging.warning(f"Invalid years value '{years}' received for weight assignment. Assigning weight 0.")
        return 0
    years = int(round(years))

    if years > 15:
        logging.info(f"Assigning experience weight 3 for {years} years.")
        return 3
    elif 8 <= years <= 15:
        logging.info(f"Assigning experience weight 2 for {years} years.")
        return 2
    elif 1 <= years <= 7:
        logging.info(f"Assigning experience weight 1 for {years} years.")
        return 1
    else:
        logging.info(f"Assigning experience weight 0 for {years} years.")
        return 0

# Cell 9: OPTIMIZED - Combined function to analyze skill weights and skill gaps in a single API call (Unchanged Logic)
def analyze_job_requirements(job_role, required_skills, cv_skills):
    """
    Analyzes skill relevance (weights) and identifies missing skills using Gemini.
    """
    if GOOGLE_API_KEY in ["YOUR_API_KEY_HERE"]:
         logging.error("GOOGLE_API_KEY is not set (using generic placeholder). Cannot call Gemini for job analysis.")
         st.error("Google API Key is not configured correctly (using generic placeholder). Cannot analyze skill weights and gaps using Gemini.")
         return {skill: 1 for skill in required_skills}, required_skills

    if not required_skills:
        logging.warning("No required skills provided for analysis. Returning empty results.")
        return {}, []
    if not cv_skills:
        logging.warning("No CV skills provided for analysis. All required skills will be marked as missing.")
        weights = {skill: 1 for skill in required_skills}
        return weights, required_skills

    try:
        # GenAI configured at start
        model = genai.GenerativeModel("gemini-1.5-flash")
        num_skills = len(required_skills)
        max_weight = max(5, num_skills // 2)

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
            f"Perform a case-insensitive comparison. Consider semantic similarity and common variations (e.g., 'Machine Learning' matches 'ML', 'AWS' matches 'Amazon Web Services', 'Python Programming' matches 'Python', 'SQL' matches 'Database Management' ONLY IF context implies database querying). Be conservative with semantic matching; if unsure, list the skill as missing.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return a single, valid JSON object containing exactly two keys:\n"
            f"- 'weights': An object where keys are the required skill strings (exactly as provided in required_skills list) and values are their assigned integer weights.\n"
            f"- 'missing_skills': An array of strings, listing only the required skills (exactly as provided in required_skills list) identified as missing from the CV skills.\n\n"
            f"Example JSON Output:\n"
            f"{{\"weights\": {{\"Python\": {max_weight}, \"Machine Learning\": {max_weight-1}, \"SQL\": {max_weight // 2}, \"Communication\": 1}}, \"missing_skills\": [\"SQL\"]}}\n\n"
            f"IMPORTANT: Ensure the output is ONLY the JSON object, with no surrounding text, comments, or markdown formatting like ```json ... ```."
        )

        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        logging.info(f"Gemini raw job analysis response (first 100 chars): {raw_response[:100]}...")

        cleaned_response = re.sub(r"^```json\s*", "", raw_response)
        cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            analysis_data = json.loads(cleaned_response)
            if not isinstance(analysis_data, dict):
                raise ValueError("Parsed JSON is not a dictionary.")
            if "weights" not in analysis_data or "missing_skills" not in analysis_data:
                 logging.error(f"Gemini response missing required keys ('weights', 'missing_skills'). Response: {cleaned_response}")
                 st.warning(f"Gemini response for job analysis was malformed (missing keys). Using default weights (1) and assuming all required skills are missing.")
                 return {skill: 1 for skill in required_skills}, required_skills

            weights = analysis_data.get("weights", {})
            missing_skills = analysis_data.get("missing_skills", [])

            if not isinstance(weights, dict):
                 logging.warning(f"Expected 'weights' to be a dict, got {type(weights)}. Using fallback weights.")
                 st.warning(f"Gemini response for job analysis had invalid format for 'weights'. Using default weight (1) for all skills.")
                 weights = {skill: 1 for skill in required_skills}

            if not isinstance(missing_skills, list):
                logging.warning(f"Expected 'missing_skills' to be a list, got {type(missing_skills)}. Using fallback.")
                st.warning(f"Gemini response for job analysis had invalid format for 'missing_skills'. Assuming all required skills are missing.")
                missing_skills = list(missing_skills) if isinstance(missing_skills, (tuple, set)) else required_skills

            final_weights = {}
            for skill in required_skills:
                if skill not in weights:
                    logging.warning(f"Skill '{skill}' missing from weights response. Assigning default weight 1.")
                    final_weights[skill] = 1
                elif not isinstance(weights[skill], int):
                     logging.warning(f"Weight for skill '{skill}' is not an integer ({type(weights[skill])}: {weights[skill]}). Setting to 1.")
                     final_weights[skill] = 1
                elif weights[skill] < 1 or weights[skill] > max_weight:
                     logging.warning(f"Weight for skill '{skill}' ({weights[skill]}) is outside the expected range [1, {max_weight}]. Clamping to range.")
                     final_weights[skill] = max(1, min(weights[skill], max_weight))
                else:
                     final_weights[skill] = weights[skill]

            validated_missing = [ms for ms in missing_skills if ms in required_skills]
            if len(validated_missing) != len(missing_skills):
                missed_original = set(missing_skills)
                missed_validated = set(validated_missing)
                extra_reported = missed_original - missed_validated
                logging.warning(f"Gemini reported missing skills not in the original required list: {extra_reported}. Filtering them out.")
                missing_skills = validated_missing

            logging.info(f"Analyzed weights for {len(final_weights)} skills and identified {len(missing_skills)} missing skills for role '{job_role}'")
            return final_weights, missing_skills
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Gemini JSON response for job analysis. Error: {str(e)}. Response: {cleaned_response[:200]}")
            st.warning(f"Failed to parse Gemini JSON response for job analysis. Using default weights (1) and assuming all required skills are missing.")
            return {skill: 1 for skill in required_skills}, required_skills
        except ValueError as ve:
            logging.error(f"Data validation error in job analysis: {ve}. Response: {cleaned_response}")
            st.warning(f"Data validation error processing Gemini response for job analysis: {ve}. Using default weights (1) and assuming all required skills are missing.")
            return {skill: 1 for skill in required_skills}, required_skills

    except Exception as e:
        logging.error(f"Error during Gemini API call for job analysis: {str(e)}")
        st.error(f"Error calling Gemini API for job analysis: {str(e)}. Check API key and network.")
        return {skill: 1 for skill in required_skills}, required_skills

# Cell 10: Updated CV Ranking Function using optimized API calls (Unchanged Logic)
def rank_cvs(cv_folder, job_role):
    """
    Ranks CVs in a folder based on skills match and relevant experience for a specific job role.
    Returns ranked list and dictionary of CV texts. Handles errors internally and logs/warns via st.
    """
    job_collection = None
    required_skills = []
    try:
        job_collection = get_job_collection()
        if job_collection is None:
             raise ValueError("Database connection failed.")

        required_skills = get_required_skills(job_collection, job_role)
        # Warnings/errors handled inside get_required_skills

    except ValueError as ve:
        logging.error(f"Setup error for ranking: {str(ve)}")
        raise ve

    cv_results = []
    cv_texts = {}
    processed_files = 0
    error_files = 0

    try:
        pdf_files_found = [f for f in os.listdir(cv_folder) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        st.error(f"The specified CV folder path does not exist: {cv_folder}")
        return [], {}
    except Exception as e:
        st.error(f"Error accessing the CV folder '{cv_folder}': {e}")
        return [], {}

    if not pdf_files_found:
        logging.warning(f"No PDF files found in the specified folder: {cv_folder}")
        st.warning(f"No PDF files found in the folder: {cv_folder}")
        return [], {}

    logging.info(f"Found {len(pdf_files_found)} PDF files to process.")
    st.info(f"Found {len(pdf_files_found)} PDF files. Starting analysis...")

    progress_bar = st.progress(0)
    total_files = len(pdf_files_found)

    for i, cv_file in enumerate(pdf_files_found):
        cv_path = os.path.join(cv_folder, cv_file)
        logging.info(f"--- Processing CV file: {cv_file} ---")
        try:
            cv_text = extract_cv_text(cv_path)

            if cv_text is None:
                error_files += 1
                logging.warning(f"Skipping '{cv_file}' due to critical text extraction error.")
                continue
            if not cv_text.strip():
                error_files += 1
                logging.warning(f"Skipping '{cv_file}' because no text could be extracted.")
                # No score possible, don't add to results but count as error/skipped
                continue # Skip to next file

            cv_texts[cv_file] = cv_text

            cv_skills, years = extract_cv_data_with_gemini(cv_text, job_role)

            if not cv_skills and years == 0:
                logging.warning(f"No skills or relevant experience years extracted from '{cv_file}' by Gemini. Ranking scores may be low.")

            exp_weight = assign_experience_weight(years)

            skill_weights = {}
            missing_skills = list(required_skills)
            skill_score = 0
            matched_skills_list = []

            if required_skills:
                skill_weights, missing_skills = analyze_job_requirements(job_role, required_skills, cv_skills)
                matched_skills_set = set(required_skills) - set(missing_skills)
                matched_skills_list = sorted(list(matched_skills_set))
                skill_score = sum(skill_weights.get(skill, 0) for skill in matched_skills_list)
            else:
                logging.info(f"No required skills defined for '{job_role}', skill score set to 0 for '{cv_file}'.")


            num_required = len(required_skills)
            num_matched = len(matched_skills_list)
            skill_match_percent = (num_matched / num_required * 100) if num_required > 0 else 100.0

            total_score = exp_weight + skill_score

            cv_results.append({
                "cv_file": cv_file,
                "years_experience": years,
                "experience_weight": exp_weight,
                "cv_skills": cv_skills,
                "required_skills": required_skills,
                "matched_skills": matched_skills_list,
                "missing_skills": missing_skills,
                "skill_score": skill_score,
                "skill_weights": skill_weights,
                "skill_match_percent": round(skill_match_percent, 1),
                "total_score": total_score
            })
            logging.info(f"Successfully processed '{cv_file}': Score = {total_score}, Relevant Years = {years}, Match = {round(skill_match_percent, 1)}%")
            processed_files += 1

        except Exception as e:
            logging.error(f"Unexpected critical error processing '{cv_file}': {str(e)}", exc_info=True)
            st.error(f"Unexpected error processing '{cv_file}': {str(e)}. It will be skipped.")
            error_files += 1
            cv_texts.pop(cv_file, None)
            continue
        finally:
             progress = (i + 1) / total_files
             progress_bar.progress(progress)


    ranked_cvs = sorted(cv_results, key=lambda x: x["total_score"], reverse=True)

    logging.info(f"--- CV Ranking Complete for role '{job_role}' ---")
    logging.info(f"Successfully processed: {processed_files} CVs")
    logging.info(f"Skipped due to errors: {error_files} CVs")
    logging.info(f"Total PDF files found: {len(pdf_files_found)}")
    logging.info(f"Total CVs ranked: {len(ranked_cvs)}")

    st.success(f"Analysis complete! Ranked {len(ranked_cvs)} CVs. Processed: {processed_files}, Skipped: {error_files}.")
    if not ranked_cvs and pdf_files_found:
        st.warning("No CVs were successfully ranked, though PDF files were found. Check logs and potential errors above.")
    elif not ranked_cvs and not pdf_files_found:
         st.warning("No PDF CVs found in the folder to rank.")

    return ranked_cvs, cv_texts

# --- START OF STREAMLIT APP ---

@st.cache_resource
def get_llm():
    if GOOGLE_API_KEY in ["YOUR_API_KEY_HERE"]:
        logging.error("FATAL: GOOGLE_API_KEY is not set (generic placeholder). Chatbot will not function.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        logging.info("LangChain Gemini LLM initialized successfully.")
        return llm
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize LangChain Gemini LLM: {str(e)}")
        st.error(f"Failed to initialize Gemini LLM for Chatbot. Chatbot will not function.\nError: {str(e)}")
        return None

def ask_cv_question_streamlit(llm, cv_filename, question, selected_cv_analysis, cv_texts, job_role):
    """Handles chatbot query using Streamlit context."""
    if not llm:
        return "Chatbot is not available due to an initialization error."
    if not cv_filename or cv_filename == "Select CV":
        return "Please select a valid CV from the dropdown first."
    if not question:
        return "Please enter a question."

    try:
        cv_text = cv_texts.get(cv_filename, "CV text not found for the selected file.")

        if selected_cv_analysis:
            analysis_summary_dict = {
                "cv_file": selected_cv_analysis.get("cv_file"),
                "total_score": selected_cv_analysis.get("total_score"),
                "years_experience": selected_cv_analysis.get("years_experience"),
                "skill_match_percent": selected_cv_analysis.get("skill_match_percent"),
                "matched_skills": selected_cv_analysis.get("matched_skills", []),
                "missing_skills": selected_cv_analysis.get("missing_skills", []),
            }
            max_list_len = 10
            if len(analysis_summary_dict["matched_skills"]) > max_list_len:
                analysis_summary_dict["matched_skills"] = analysis_summary_dict["matched_skills"][:max_list_len] + ["..."]
            if len(analysis_summary_dict["missing_skills"]) > max_list_len:
                 analysis_summary_dict["missing_skills"] = analysis_summary_dict["missing_skills"][:max_list_len] + ["..."]
            analysis_summary = json.dumps(analysis_summary_dict, indent=2)
        else:
            analysis_summary = "Analysis data not found for this CV."
            logging.warning(f"Analysis data not found for chatbot query on CV: {cv_filename}")


        system_prompt = (
            "You are a helpful AI assistant expert in recruitment and CV analysis. "
            "Answer the user's question based *strictly* on the provided CV text and its analysis summary. "
            "Be concise and factual. If the information isn't in the provided text or summary, state that explicitly (e.g., 'The CV text does not mention...'). "
            "Do not make assumptions or provide information beyond the context given. Do not browse external websites or access external knowledge."
        )
        human_prompt_content = (
            f"Context:\n"
            f"Job Role Context for Analysis: {job_role}\n"
            f"Selected CV Filename: {cv_filename}\n\n"
            f"CV Analysis Summary (use this for scores, missing skills etc.):\n```json\n{analysis_summary}\n```\n\n"
            f"Full CV Text (use this for details about projects, experience descriptions etc.):\n```text\n{cv_text}\n```\n\n"
            f"User Question: {question}"
        )
        answer_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt_content)
        ]

        logging.info(f"Sending question to LLM for CV '{cv_filename}': {question}")
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
        logging.info(f"LLM response received (length: {len(answer)}).")
        return answer

    except Exception as e:
        logging.error(f"Error processing chatbot query: {str(e)}", exc_info=True)
        st.error(f"Sorry, an error occurred while processing your question: {str(e)}")
        return f"Sorry, an error occurred while processing your question: {str(e)}"

def main():
    st.set_page_config(layout="wide", page_title="CV Ranking System")
    st.title("📄 CV Ranking System (Gemini & MongoDB)")

    # --- Configuration Checks ---
    # Check for *generic* placeholders first - this is a fatal error
    if _MISSING_CONFIG:
        st.error(f"**Configuration Error:** The following configurations are missing or using generic placeholders: {', '.join(_MISSING_CONFIG)}. Please set them correctly in the script.")
        st.stop() # Stop if generic placeholders are found

    # Check if the *specific default/hardcoded* keys are being used and show a warning
    if _USING_DEFAULTS:
        st.warning(f"**Security Warning:** Using default hardcoded credentials for: {', '.join(_USING_DEFAULTS)}. Move these to environment variables or Streamlit secrets for production.")

    # --- Initialize Session State ---
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'cv_texts' not in st.session_state:
        st.session_state.cv_texts = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_cv_for_chat' not in st.session_state:
        st.session_state.selected_cv_for_chat = None
    if 'analysis_job_role' not in st.session_state:
        st.session_state.analysis_job_role = None

    # --- Load Job Roles ---
    job_roles_list = get_job_roles_from_db()
    is_role_list_valid = not (job_roles_list[0].startswith("Error") or job_roles_list[0].startswith("No job roles"))

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Analysis Configuration")
        cv_folder = st.text_input("Enter Full Path to CV Folder:", key="cv_folder_input", help="Provide the absolute path to the folder containing PDF CVs.")

        if is_role_list_valid:
            selected_job_role = st.selectbox("Select Target Job Role:", job_roles_list, key="job_role_select")
        else:
            st.error(f"Job Roles: {job_roles_list[0]}")
            selected_job_role = None
            st.selectbox("Select Target Job Role:", [job_roles_list[0]], disabled=True)

        run_button = st.button("Run Analysis", key="run_analysis_button", disabled=not is_role_list_valid)

    # --- Main Area ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📊 Analysis Results")

        if run_button:
            st.session_state.results = None
            st.session_state.cv_texts = {}
            st.session_state.messages = []
            st.session_state.selected_cv_for_chat = None
            st.session_state.analysis_job_role = None

            validation_passed = True
            if not cv_folder:
                st.error("Please enter the path to the CV folder.")
                validation_passed = False
            elif not os.path.isdir(cv_folder):
                 st.error(f"The provided path is not a valid directory: {cv_folder}")
                 validation_passed = False
            if not selected_job_role or selected_job_role == "Select Job Role":
                st.error("Please select a valid job role.")
                validation_passed = False

            if validation_passed:
                st.info(f"Starting analysis for job role: '{selected_job_role}' in folder: '{cv_folder}'")
                try:
                    with st.spinner("Analyzing CVs... This may take a few minutes."):
                        ranked_cvs, cv_texts_data = rank_cvs(cv_folder, selected_job_role)
                        st.session_state.results = ranked_cvs
                        st.session_state.cv_texts = cv_texts_data
                        st.session_state.analysis_job_role = selected_job_role
                        st.session_state.messages = []
                        st.session_state.selected_cv_for_chat = None
                except ValueError as ve:
                     st.error(f"Analysis Setup Error: {str(ve)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {str(e)}")
                    logging.error(f"Analysis failed critically: {str(e)}", exc_info=True)

        if st.session_state.results is not None:
            if st.session_state.results:
                st.subheader(f"Ranked CVs for Role: {st.session_state.analysis_job_role}")
                for i, cv_data in enumerate(st.session_state.results):
                    rank = i + 1
                    with st.expander(f"Rank {rank}: {cv_data['cv_file']} (Score: {cv_data['total_score']})", expanded=(rank <= 3)):
                        st.markdown(f"**Overall Score:** {cv_data['total_score']}")
                        st.markdown(f"**Relevant Experience:** {cv_data['years_experience']} years (Weight: {cv_data['experience_weight']})")
                        st.markdown(f"**Skill Match:** {cv_data['skill_match_percent']}% (Score: {cv_data['skill_score']})")

                        max_skills_display = 15
                        matched_skills_display = cv_data['matched_skills']
                        missing_skills_display = cv_data['missing_skills']
                        cv_skills_display = cv_data['cv_skills']

                        if len(matched_skills_display) > max_skills_display: matched_skills_display = matched_skills_display[:max_skills_display] + ['...']
                        if len(missing_skills_display) > max_skills_display: missing_skills_display = missing_skills_display[:max_skills_display] + ['...']
                        if len(cv_skills_display) > max_skills_display: cv_skills_display = cv_skills_display[:max_skills_display] + ['...']

                        st.markdown(f"**Matched Skills ({len(cv_data['matched_skills'])}):**")
                        st.markdown(f"> _{', '.join(matched_skills_display) if matched_skills_display else 'None'}_")
                        st.markdown(f"**Missing Skills ({len(cv_data['missing_skills'])}):**")
                        st.markdown(f"> _{', '.join(missing_skills_display) if missing_skills_display else 'None'}_")

            elif 'analysis_job_role' in st.session_state and st.session_state.analysis_job_role:
                 st.warning(f"No CVs were successfully ranked for the role '{st.session_state.analysis_job_role}'. Check the folder content and logs if files were expected.")
        else:
            st.info("Select a folder and job role in the sidebar, then click 'Run Analysis' to see results.")

    with col2:
        st.header("💬 Chatbot")
        llm = get_llm()

        if llm is None:
            st.error("Chatbot cannot function because the Language Model failed to initialize. Check API Key configuration.")
        elif st.session_state.results is None:
             st.info("Run an analysis first to enable the chatbot.")
        elif not st.session_state.results:
            st.warning("No CVs were ranked. Chatbot is unavailable.")
        else:
            cv_files_ranked = ["Select CV"] + [res["cv_file"] for res in st.session_state.results]
            selected_cv_filename = st.selectbox(
                "Select CV to ask about:",
                cv_files_ranked,
                key='chatbot_cv_select',
                index=0
            )

            if selected_cv_filename != "Select CV":
                st.session_state.selected_cv_for_chat = selected_cv_filename
            else:
                st.session_state.selected_cv_for_chat = None

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about the selected CV...", disabled=(st.session_state.selected_cv_for_chat is None)):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                selected_cv_analysis_data = next((item for item in st.session_state.results if item["cv_file"] == st.session_state.selected_cv_for_chat), None)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Thinking..."):
                        response = ask_cv_question_streamlit(
                            llm,
                            st.session_state.selected_cv_for_chat,
                            prompt,
                            selected_cv_analysis_data,
                            st.session_state.cv_texts,
                            st.session_state.analysis_job_role
                        )
                    message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            elif st.session_state.selected_cv_for_chat is None and len(st.session_state.results or []) > 0:
                 st.info("Select a CV from the dropdown above to start chatting.")


# Cell 12: Run the application (using Streamlit)
if __name__ == "__main__":
    main()
# --- END OF FILE Section3_for_recruitment_testq_2_streamlit_hardcoded.py ---