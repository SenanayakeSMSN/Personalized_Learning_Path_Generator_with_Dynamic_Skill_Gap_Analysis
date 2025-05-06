

"""
Streamlit version for batch CV processing and ranking.
Processes CVs in a folder, stores data in MongoDB, ranks based on job role.
Includes a Gemini-powered chatbot for querying processed CV data.
"""

# Cell 2: Import Libraries and Setup
import streamlit as st
import os
import json
import logging
import re
from datetime import datetime
from pymongo import MongoClient, errors as mongo_errors
from PIL import Image # Keep for pytesseract if used
import concurrent.futures
import uuid
import time
import traceback # For detailed error logging
import pandas as pd # For displaying ranked results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - S3 - %(levelname)s - %(message)s')

# --- IMPORTANT CONFIGURATION ---
# Use Streamlit secrets for production is recommended, but sticking to hardcoded for now as requested.
# --- START SENSITIVE ---
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo" # Replace with your actual Google API Key
MONGODB_URI = "mongodb+srv://akilapremarathna0:123@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority" # Replace with your actual MongoDB connection string
# --- END SENSITIVE ---
TARGET_DB_NAME = "skillgapanalysis" # Should match the DB in your MONGODB_URI
TARGET_COLLECTION_NAME = "cv_extracted_data" # Collection for extracted data
JOB_ROLE_COLLECTION_NAME = "jobrole_skill" # Collection containing job roles and required skills
# --- End Configuration ---

# --- Dependency Imports and Checks ---
# Assuming pdfplumber, google-generativeai, pymongo, Pillow, pytesseract installed
try:
    import pdfplumber
except ImportError:
    st.error("pdfplumber is not installed. Please install it (`pip install pdfplumber`).")
    st.stop()
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_API_KEY":
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            logging.info("Google Generative AI configured.")
        except Exception as e:
             logging.warning(f"Could not configure Google API key: {e}. Calls might fail.")
             st.warning(f"Could not configure Google API key: {e}. Chatbot and CV analysis might fail.")
             # Don't stop the app yet, maybe user just wants to view old results?
    else:
        logging.warning("GOOGLE_API_KEY is not set or is a placeholder. Gemini calls will fail.")
        st.error("GOOGLE_API_KEY is not set correctly in the script. Gemini features (Analysis & Chatbot) will fail.")
except ImportError:
    st.error("google-generativeai is not installed. Please install it (`pip install google-generativeai`).")
    st.stop()
try:
    import pytesseract
    # Add path if needed, typically set system-wide or via ENV VAR
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    logging.warning("pytesseract not found. OCR fallback disabled. Install with `pip install pytesseract` and ensure Tesseract engine is installed.")
except Exception as e:
    logging.warning(f"Could not configure pytesseract path: {e}. Ensure Tesseract is installed and in your PATH.")

# --- Constants ---
MAX_WORKERS = 10 # For concurrent processing
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Model for CV analysis and Chatbot

# --- Helper Functions ---
def safe_json_loads(text):
    """Safely cleans and parses JSON from text, handling markdown code blocks."""
    if not text: return None
    try:
        # More robust cleaning
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
             cleaned_text = cleaned_text[3:] # Handle cases like ``` {} ```
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        if not cleaned_text: return None
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}. Text was: {text[:200]}...")
        # Try a more aggressive cleanup (remove potential leading/trailing text)
        match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                 logging.error("Aggressive JSON cleanup also failed.")
                 return None
        return None
    except Exception as e:
        logging.error(f"Unexpected error during JSON parsing: {e}. Text was: {text[:200]}...")
        return None

# --- MongoDB Connection (Cached) ---
@st.cache_resource(ttl=3600)
def connect_to_mongodb_resource():
    """Connects to MongoDB, returns client, db, and collections."""
    if "YOUR_MONGODB_URI" in MONGODB_URI or "username:password" in MONGODB_URI or not MONGODB_URI:
         logging.error("MONGODB_URI is not set correctly.")
         st.error("MongoDB URI is not configured correctly in the script.")
         st.stop()
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Ping to verify connection early
        client.admin.command('ping')
        logging.info(f"Successfully connected to MongoDB cluster.")
        db = client[TARGET_DB_NAME]
        extracted_data_collection = db[TARGET_COLLECTION_NAME]
        job_role_collection = db[JOB_ROLE_COLLECTION_NAME]

        # Check if collections exist (optional, provides feedback)
        coll_names = db.list_collection_names()
        if TARGET_COLLECTION_NAME not in coll_names:
             logging.warning(f"Collection '{TARGET_COLLECTION_NAME}' not found. It will be created on first insert.")
        if JOB_ROLE_COLLECTION_NAME not in coll_names:
             logging.warning(f"Collection '{JOB_ROLE_COLLECTION_NAME}' not found. Fetching job roles will fail.")
             # Don't stop here, maybe user wants to process first

        # Indexes should be created manually or via an admin script, not here.
        logging.info(f"Using DB '{TARGET_DB_NAME}', Collection (Extracted): '{TARGET_COLLECTION_NAME}', Collection (Job Roles): '{JOB_ROLE_COLLECTION_NAME}'")
        return client, db, extracted_data_collection, job_role_collection
    except mongo_errors.ConnectionFailure as e:
        logging.error(f"MongoDB Connection Failure: {e}.")
        st.error(f"Failed to connect to MongoDB: {e}. Please check URI and network access.")
        st.stop()
    except Exception as e:
        logging.error(f"MongoDB Setup Error: {str(e)}")
        st.error(f"Error setting up MongoDB connection: {str(e)}")
        st.stop()

# --- Get Job Roles (Cached Data) ---
@st.cache_data(ttl=3600)
def get_job_roles_data():
    """Fetches job roles, designed to work with cached connection resource."""
    client = None
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        db = client[TARGET_DB_NAME]
        job_role_collection = db[JOB_ROLE_COLLECTION_NAME]
        if JOB_ROLE_COLLECTION_NAME not in db.list_collection_names():
             return ["Error: Job Role Collection not found"]

        roles_cursor = job_role_collection.find({}, {"Job_Role": 1})
        roles = [doc["Job_Role"] for doc in roles_cursor if "Job_Role" in doc and doc["Job_Role"]]
        return sorted(list(set(roles)))
    except Exception as e:
        logging.error(f"Failed to fetch job roles for data cache: {str(e)}")
        return [f"Error loading roles: {e}"]
    finally:
        if client:
            client.close()


# Cell 4: Enhanced CV Text Extraction (Function logic unchanged)
def extract_cv_text(cv_path):
    """Extracts text from a PDF CV, attempting OCR as a fallback. Returns None on failure."""
    text = ""
    filename = os.path.basename(cv_path)
    try:
        with pdfplumber.open(cv_path) as pdf:
            logging.debug(f"Processing PDF: {filename}")
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n\n"
                else:
                    # Try OCR only if pdfplumber failed and pytesseract is available/configured
                    if 'pytesseract' in globals() and hasattr(pytesseract, 'image_to_string'):
                        logging.info(f"No text from page {page_number} via pdfplumber for {filename}. Attempting OCR.")
                        try:
                            img = page.to_image(resolution=300).original # Good resolution for OCR
                            ocr_text = pytesseract.image_to_string(img, lang='eng') # Specify language if known
                            if ocr_text and ocr_text.strip():
                                text += ocr_text + "\n\n"
                                logging.debug(f"Extracted OCR text from page {page_number} for {filename}")
                            else:
                                text += "\n\n" # Add space even if OCR yields nothing
                        except ImportError:
                             logging.error("OCR skipped: pytesseract/PIL not installed correctly.")
                             text += "\n\n"
                        except Exception as ocr_e:
                            if "Tesseract is not installed or" in str(ocr_e):
                                logging.error("OCR failed: Tesseract not found. Install/configure Tesseract.")
                            else:
                                logging.error(f"OCR failed for page {page_number} in {filename}: {str(ocr_e)}.")
                            text += "\n\n"
                    else:
                        logging.warning(f"OCR skipped for page {page_number} in {filename}: pytesseract not available or configured.")
                        text += "\n\n" # Maintain spacing

            if not text.strip():
                logging.warning(f"No text extracted from CV '{filename}'.")
                return "" # Return empty string for no text, None for errors below
            logging.debug(f"Extracted text from CV '{filename}'. Length: {len(text)} chars.")
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error(f"PDF '{filename}' is password-protected.")
        return "Error: PDF Password Protected"
    except FileNotFoundError:
        logging.error(f"CV file not found: {cv_path}")
        return "Error: File Not Found"
    except Exception as e:
        logging.error(f"Error reading CV '{filename}': {str(e)}")
        return f"Error: Reading Failed - {str(e)[:50]}" # Return error message


# Cell 6: Gemini Data Extraction (Function logic unchanged)
# Ensure GOOGLE_API_KEY is configured before calling
def extract_cv_data_with_gemini(cv_text, job_role, cv_filename="Unknown CV"):
    """
    Extracts skills and relevant years of experience using Gemini.
    Returns (skills, years) or (None, None) on failure.
    Includes basic retry on API errors, but not key rotation here.
    """
    error_result = None, None

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY":
         logging.error("GOOGLE_API_KEY is not set. Cannot call Gemini for CV analysis.")
         return error_result

    if not cv_text or not cv_text.strip():
        logging.error(f"CV text is empty for '{cv_filename}', cannot extract data.")
        return error_result
    if not job_role or not job_role.strip():
        logging.error(f"Job role context is empty for '{cv_filename}', cannot extract relevant experience.")
        return error_result

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        current_year = datetime.now().year
        current_month = datetime.now().month
        prompt = (
            f"You are an expert in CV analysis for the specific job role of '{job_role}'. "
            f"Analyze the following CV text for '{cv_filename}' and extract two pieces of information:\n\n"
            f"1.  **SKILLS:** Identify all technical skills (e.g., programming languages, frameworks, tools, software, methodologies) "
            f"and relevant soft skills (e.g., communication, leadership, teamwork, problem-solving) mentioned or clearly implied in the text. "
            f"Consolidate variations (e.g., 'Machine Learning' and 'ML' should both result in 'Machine Learning'). "
            f"Present the skills as a list of unique strings.\n\n"
            f"2.  **RELEVANT EXPERIENCE (Years):** Calculate the total duration of professional work experience that is directly relevant to the '{job_role}' position. "
            f"Consider job titles, responsibilities described, and skills used in each role mentioned in the CV. "
            f"Exclude internships, academic projects, or roles clearly unrelated to '{job_role}'. "
            f"Follow these rules for calculation:\n"
            f"    *   Parse dates (e.g., MM/YYYY, Month YYYY, YYYY-YYYY). Use {current_month}/{current_year} for 'Present' or 'Current'. Assume Jan/Dec for missing months.\n"
            f"    *   Calculate duration for each relevant job in months: `(end_year - start_year) * 12 + (end_month - start_month)`.\n"
            f"    *   Sum the total months across all *relevant* jobs.\n"
            f"    *   Convert the total months to years by dividing by 12. Round the result to the nearest whole integer.\n"
            f"    *   If no relevant experience is found, return 0.\n\n"
            f"**Output Format:** Return ONLY a single, valid JSON object containing exactly two keys:\n"
            f"   - `\"skills\"`: An array of strings, where each string is a unique skill identified.\n"
            f"   - `\"years\"`: An integer representing the total years of relevant experience (rounded).\n\n"
            f"**Example Output:**\n"
            f"{{\"skills\": [\"Python\", \"Machine Learning\", \"SQL\", \"Data Analysis\", \"Communication\"], \"years\": 5}}\n\n"
            f"**CV Text:**\n"
            f"```text\n{cv_text[:15000]}\n```" # Limit text size
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)

                if not response.parts:
                     block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
                     block_message = response.prompt_feedback.block_reason_message if response.prompt_feedback else 'No message'
                     logging.error(f"Gemini response blocked for {cv_filename}. Reason: {block_reason} - {block_message}")
                     return error_result

                raw_response = response.text.strip()
                logging.debug(f"Gemini raw response for {cv_filename}: {raw_response[:100]}...")

                cv_data = safe_json_loads(raw_response)

                if cv_data is None or not isinstance(cv_data, dict) or "skills" not in cv_data or "years" not in cv_data:
                    logging.error(f"Invalid or missing JSON data from Gemini for {cv_filename}. Response: {raw_response[:200]}")
                    skills_fallback = re.findall(r'"(.*?)"', raw_response)
                    years_match = re.search(r'["\']years["\']\s*:\s*(\d+)', raw_response)
                    years_fallback = int(years_match.group(1)) if years_match else 0
                    if skills_fallback or years_fallback > 0:
                         logging.warning(f"Using fallback parsing for {cv_filename}.")
                         return skills_fallback, years_fallback
                    if attempt < max_retries - 1:
                        logging.warning(f"Retrying Gemini call for {cv_filename} due to bad JSON format...")
                        time.sleep(3)
                        continue
                    else:
                        return error_result

                skills = cv_data.get("skills", [])
                years_raw = cv_data.get("years", 0)

                if not isinstance(skills, list):
                    logging.warning(f"Skills data is not a list for {cv_filename}, received: {type(skills)}. Setting to empty list.")
                    skills = []
                else:
                    skills = [str(s) for s in skills if isinstance(s, (str, int, float))]

                if not isinstance(years_raw, int):
                    try:
                        years = int(float(years_raw))
                    except (ValueError, TypeError):
                        logging.warning(f"Years data is not a valid integer for {cv_filename}, received: {years_raw}. Setting to 0.")
                        years = 0
                else:
                    years = years_raw

                logging.info(f"Successfully extracted {len(skills)} skills, {years} relevant years for '{job_role}' from {cv_filename}")
                return skills, years

            except Exception as api_error:
                logging.error(f"Gemini API call error for {cv_filename} (Attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                if attempt < max_retries - 1:
                    time.sleep(5 + attempt * 5)
                    logging.info(f"Retrying API call for {cv_filename}...")
                    continue
                else:
                    logging.error(f"Max retries reached for Gemini API call for {cv_filename}.")
                    return error_result

        return error_result

    except Exception as e:
        logging.error(f"Unexpected error during Gemini CV data extraction setup for {cv_filename}: {str(e)}", exc_info=True)
        return error_result

# Cell 7: MongoDB Query for Required Skills (Unchanged logic, adapted for Streamlit context)
def get_required_skills_from_db(job_role, job_role_collection):
    """Fetches and cleans the list of required skills using provided collection handle."""
    required_skills = []
    try:
        job_doc = job_role_collection.find_one({"Job_Role": {"$regex": f"^{re.escape(job_role)}$", "$options": "i"}})

        if not job_doc:
            logging.error(f"Job role '{job_role}' not found in collection '{job_role_collection.name}'.")
            return None

        required_skills_raw = job_doc.get("Required_Skills")
        if not required_skills_raw or not isinstance(required_skills_raw, str):
             logging.warning(f"'Required_Skills' missing or not a string for job role '{job_role}'.")
             return []

        required_skills = [skill.strip().lower() for skill in required_skills_raw.split(",") if skill.strip()]

        if not required_skills:
            logging.warning(f"No valid required skills derived for job role '{job_role}'. Raw value was: '{required_skills_raw}'")
            return []

        logging.info(f"Fetched {len(required_skills)} required skills for {job_role}.")
        return required_skills

    except mongo_errors.PyMongoError as e:
         logging.error(f"MongoDB error retrieving required skills for '{job_role}': {e}")
         st.error(f"Database error fetching skills for {job_role}: {e}")
         return None
    except Exception as e:
        logging.error(f"Unexpected error retrieving required skills for '{job_role}': {str(e)}")
        st.error(f"Unexpected error fetching skills for {job_role}: {e}")
        return None

# Cell 8: Experience Weight Assignment (Unchanged logic)
def assign_experience_weight(years):
    """Assigns a weight based on years of relevant experience."""
    if not isinstance(years, (int, float)) or years < 0:
        return 0
    years = int(round(years))
    if years >= 15: return 3
    elif 8 <= years < 15: return 2
    elif 1 <= years < 8: return 1
    else: return 0

# Cell 9: Get Skill Weights from Gemini (Cached per Job Role for duration of app run)
@st.cache_data(ttl=1800) # Cache weights for 30 mins per job role/skills combo
def get_skill_weights_for_job_role(_job_role, _required_skills): # Underscore to indicate cache key params
    """
    Uses Gemini to assign importance weights to required skills for a job role.
    Returns a dictionary {skill: weight} or None on failure.
    Input skills should be normalized (e.g., lowercased).
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY":
        logging.error("GOOGLE_API_KEY not set. Cannot get skill weights.")
        st.error("API Key not configured. Cannot get skill weights.")
        return None

    if not _required_skills:
        logging.warning(f"No required skills provided for '{_job_role}', cannot assign weights.")
        return {}

    error_result = None

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        num_skills = len(_required_skills)
        max_weight = min(max(3, num_skills // 3), 10)
        required_skills_json = json.dumps(_required_skills)

        prompt = (
            f"You are an expert technical recruiter specializing in the '{_job_role}' domain.\n\n"
            f"**TASK: Skill Weighting for '{_job_role}'**\n\n"
            f"Analyze the following list of essential skills required for this role:\n"
            f"{required_skills_json}\n\n"
            f"Assign an integer weight to EACH skill based on its **criticality and relevance** specifically for a candidate applying to the '{_job_role}' position. Use the following scale:\n"
            f"- 1: Foundational / Generally expected (e.g., basic office software, very common soft skills).\n"
            f"- ... (Intermediate weights for core competencies)\n"
            f"- {max_weight}: Highly Critical / Core Technical Skill (Essential for daily tasks, difficult to substitute).\n\n"
            f"Consider factors like: Is the skill niche or common? How central is it to the role's primary function? Is it a hard technical skill or a soft skill?\n\n"
            f"**OUTPUT FORMAT:**\n"
            f"Return ONLY a single, valid JSON object where:\n"
            f"- Keys are the exact required skill strings (lowercase, as provided in the input list).\n"
            f"- Values are their assigned integer weights (between 1 and {max_weight}).\n\n"
            f"**Example (if max_weight=5):**\n"
            f"{{\"python\": 5, \"machine learning\": 5, \"sql\": 4, \"data visualization\": 3, \"communication\": 2}}\n\n"
            f"**IMPORTANT:** Ensure the output is ONLY the JSON object, with no surrounding text, comments, or markdown formatting."
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)

                if not response.parts:
                     block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
                     logging.error(f"Gemini response blocked for skill weights ({_job_role}). Reason: {block_reason}")
                     return error_result

                raw_response = response.text.strip()
                logging.debug(f"Gemini raw skill weight response for {_job_role}: {raw_response[:100]}...")

                weights_data = safe_json_loads(raw_response)

                if weights_data is None or not isinstance(weights_data, dict):
                    logging.error(f"Invalid or missing JSON for skill weights from Gemini ({_job_role}). Response: {raw_response[:200]}")
                    if attempt < max_retries - 1:
                        logging.warning(f"Retrying Gemini call for skill weights ({_job_role})...")
                        time.sleep(3)
                        continue
                    else:
                         return error_result

                final_weights = {}
                valid_response = True
                for skill in _required_skills:
                    weight = weights_data.get(skill)

                    if weight is None:
                         logging.warning(f"Weight missing for required skill '{skill}' in Gemini response for {_job_role}. Assigning default weight 1.")
                         final_weights[skill] = 1
                    elif not isinstance(weight, int) or not (1 <= weight <= max_weight):
                         logging.warning(f"Invalid weight '{weight}' (type {type(weight)}) for skill '{skill}' for {_job_role}. Must be int between 1 and {max_weight}. Assigning default weight 1.")
                         final_weights[skill] = 1
                    else:
                         final_weights[skill] = weight

                missing_from_response = set(_required_skills) - set(weights_data.keys())
                if missing_from_response:
                    logging.warning(f"Gemini response did not provide weights for all required skills ({_job_role}): {missing_from_response}. Assigning default 1.")
                    for skill in missing_from_response:
                         if skill not in final_weights:
                            final_weights[skill] = 1

                if len(final_weights) != len(_required_skills):
                     logging.error(f"Weight generation mismatch for {_job_role}. Expected {len(_required_skills)} weights, got {len(final_weights)}. Weights: {final_weights}")

                logging.info(f"Successfully obtained skill weights for {len(final_weights)} skills for role '{_job_role}'.")
                return final_weights

            except Exception as api_error:
                logging.error(f"Gemini API call error during skill weight analysis for {_job_role} (Attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                if attempt < max_retries - 1:
                    time.sleep(5 + attempt * 5)
                    logging.info(f"Retrying API call for skill weights ({_job_role})...")
                    continue
                else:
                    logging.error(f"Max retries reached for Gemini skill weight call ({_job_role}).")
                    return error_result

        return error_result

    except Exception as e:
        logging.error(f"Unexpected error during skill weight setup for {_job_role}: {str(e)}", exc_info=True)
        return error_result


# Cell 10: Function to process a single CV (Focuses on extraction + storage)
def process_single_cv(cv_path, job_role, batch_id, db_collection_handle):
    """
    Processes a single CV: Extracts text, extracts data via Gemini, stores in MongoDB.
    Returns dict with status, filename, and optionally error message.
    Uses the provided MongoDB collection handle directly.
    """
    cv_filename = os.path.basename(cv_path)
    logging.info(f"[Batch:{batch_id[:6]}] Processing CV: {cv_filename} for Job: {job_role}")
    result = {"filename": cv_filename, "status": "failed", "error": None} # Default result

    try:
        # 1. Extract Text
        cv_text = extract_cv_text(cv_path)
        if cv_text is None or isinstance(cv_text, str) and cv_text.startswith("Error:"):
            error_msg = cv_text if isinstance(cv_text, str) else "Unknown text extraction error"
            logging.error(f"Text extraction failed for {cv_filename}: {error_msg}")
            result["error"] = f"Text Extraction Failed: {error_msg}"
            return result

        if not cv_text:
            logging.warning(f"No text content extracted from {cv_filename}. Skipping Gemini analysis.")
            result["error"] = "No text content found in CV"
            return result

        # 2. Extract Data via Gemini
        extracted_skills, extracted_years = extract_cv_data_with_gemini(cv_text, job_role, cv_filename)

        if extracted_skills is None or extracted_years is None:
            logging.error(f"Gemini data extraction failed for {cv_filename}.")
            result["error"] = "Gemini Data Extraction Failed"
            return result

        # 3. Store in MongoDB
        data_to_store = {
            "analysis_batch_id": batch_id,
            "cv_filename": cv_filename,
            "job_role_context": job_role,
            "extracted_skills": extracted_skills,
            "extracted_relevant_years": extracted_years,
            "processed_timestamp": datetime.utcnow(),
            # Storing raw text is generally not recommended due to size,
            # but might be needed if chatbot needs more than summary.
            # "cv_text_full": cv_text # Optional: Add if needed for chatbot context later
        }

        try:
            insert_result = db_collection_handle.insert_one(data_to_store)
            logging.info(f"Stored extracted data for {cv_filename} (ID: {insert_result.inserted_id})")
            result["status"] = "success"
            # Include the stored data ID if needed later
            result["mongo_id"] = str(insert_result.inserted_id)
            return result
        except mongo_errors.PyMongoError as e:
            logging.error(f"MongoDB insert failed for {cv_filename}: {e}")
            result["error"] = f"MongoDB Insert Failed: {e}"
            return result

    except Exception as e:
        logging.error(f"Critical error processing '{cv_filename}': {str(e)}", exc_info=True)
        result["error"] = f"Critical Processing Error: {str(e)}"
        return result


# Cell 11: Batch CV Processing Function (Focuses on extraction + storage)
def process_cvs_batch(cv_folder, job_role, extracted_data_collection):
    """
    Processes all PDF CVs in a folder in parallel, extracts data using Gemini,
    and stores the results in MongoDB with a unique batch ID.
    Returns batch_id, success_count, error_count, list_of_errors.
    """
    if not cv_folder or not os.path.isdir(cv_folder):
         st.error(f"Invalid CV folder path provided: {cv_folder}")
         return None, 0, 0, [{"filename": "N/A", "error": "Invalid folder path"}]
    if not job_role:
         st.error("Job role cannot be empty for batch processing.")
         return None, 0, 0, [{"filename": "N/A", "error": "Missing job role"}]

    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logging.info(f"--- Starting CV Batch Processing ---")
    logging.info(f"Batch ID: {batch_id}")
    logging.info(f"CV Folder: {cv_folder}")
    logging.info(f"Job Role Context: {job_role}")
    logging.info(f"Max Parallel Workers: {MAX_WORKERS}")

    pdf_files = []
    try:
        pdf_files = [os.path.join(cv_folder, f) for f in os.listdir(cv_folder)
                     if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(cv_folder, f))]
    except Exception as e:
        st.error(f"Error reading CV folder '{cv_folder}': {e}")
        return None, 0, 0, [{"filename": "N/A", "error": f"Cannot access folder: {e}"}]

    if not pdf_files:
        logging.warning(f"No PDF files found in: {cv_folder}")
        st.warning(f"No PDF files found in the specified folder: {cv_folder}")
        return batch_id, 0, 0, []

    total_files = len(pdf_files)
    logging.info(f"Found {total_files} PDF files to process.")

    success_count = 0
    error_count = 0
    error_details = []
    start_time = time.time()

    progress_bar = st.progress(0, text=f"Starting batch processing for {total_files} CVs...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_cv, pdf_path, job_role, batch_id, extracted_data_collection): pdf_path for pdf_path in pdf_files}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            pdf_path = futures[future]
            cv_filename = os.path.basename(pdf_path)
            try:
                result = future.result()
                if result and result.get("status") == "success":
                    success_count += 1
                else:
                    error_count += 1
                    error_msg = result.get("error", "Unknown processing error")
                    logging.warning(f"Processing failed for {cv_filename}: {error_msg}")
                    error_details.append({"filename": cv_filename, "error": error_msg})
            except Exception as exc:
                error_count += 1
                error_msg = f"Task execution exception: {exc}"
                logging.error(f"CV '{cv_filename}' task generated exception: {exc}", exc_info=True)
                error_details.append({"filename": cv_filename, "error": error_msg})

            progress = (i + 1) / total_files
            progress_text = f"Processing CV {i+1}/{total_files} ({cv_filename})... Success: {success_count}, Failed: {error_count}"
            progress_bar.progress(progress, text=progress_text)


    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"--- Batch Processing Complete (Batch ID: {batch_id}) ---")
    logging.info(f"Processed: {success_count}, Failed/Skipped: {error_count} in {duration:.2f} seconds")
    progress_bar.progress(1.0, text=f"Batch complete! Processed: {success_count}, Failed: {error_count} in {duration:.2f}s")

    return batch_id, success_count, error_count, error_details


# Cell 12: Function to Rank Processed CVs
def rank_processed_cvs(batch_id, job_role, extracted_data_collection, job_role_collection):
    """
    Retrieves processed CV data for a batch, scores, ranks them, and stores retrieved data in session state for chatbot.
    Returns a list of ranked CV dictionaries or None on failure.
    """
    logging.info(f"--- Starting Ranking for Batch: {batch_id}, Job Role: {job_role} ---")
    ranked_results = []

    # 1. Fetch Required Skills (Lowercase for internal use)
    required_skills_lower = get_required_skills_from_db(job_role, job_role_collection)
    if required_skills_lower is None:
        logging.error(f"Cannot proceed with ranking: Failed to get required skills for '{job_role}'.")
        # Clear context data if ranking fails early
        st.session_state.current_batch_cv_data = None
        return None
    if not required_skills_lower:
        logging.warning(f"No required skills found for '{job_role}'. Ranking will be based on experience only.")

    # 2. Get Skill Weights
    skill_weights = {}
    if required_skills_lower:
        with st.spinner(f"Getting skill importance weights for '{job_role}'..."):
             skill_weights = get_skill_weights_for_job_role(job_role, tuple(sorted(required_skills_lower)))

        if skill_weights is None:
             logging.error(f"Cannot proceed with ranking: Failed to get skill weights for '{job_role}'.")
             st.error(f"Failed to get skill weights for '{job_role}'. Ranking might be inaccurate.")
             # Clear context data if ranking fails early
             st.session_state.current_batch_cv_data = None
             return None
        elif not skill_weights:
             logging.warning(f"Received empty skill weights for '{job_role}'. Treating all skills with weight 0.")

    # 3. Retrieve Processed Data for the Batch from MongoDB
    processed_cv_data_list = [] # Initialize
    try:
        with st.spinner(f"Retrieving processed CV data for batch '{batch_id[:8]}...'..."):
             # Retrieve necessary fields for ranking AND chatbot context
             # Exclude large fields like 'cv_text_full' if stored and not needed immediately
             processed_cv_data_cursor = extracted_data_collection.find(
                 {"analysis_batch_id": batch_id},
                 {"cv_filename": 1, "extracted_skills": 1, "extracted_relevant_years": 1, "_id": 1}
             )
             processed_cv_data_list = list(processed_cv_data_cursor)

        if not processed_cv_data_list:
            logging.warning(f"No processed CV data found in MongoDB for batch_id: {batch_id}")
            st.warning(f"No processed CV data found for batch '{batch_id}'. Cannot rank or chat.")
            st.session_state.current_batch_cv_data = [] # Store empty list
            return [] # Return empty list for ranking

        logging.info(f"Retrieved {len(processed_cv_data_list)} processed CV records for ranking.")
        # Store the retrieved data (summary) in session state for the chatbot
        # Convert MongoDB _id to string for JSON serialization if needed later
        for item in processed_cv_data_list:
            item['_id'] = str(item['_id'])
        st.session_state.current_batch_cv_data = processed_cv_data_list

    except mongo_errors.PyMongoError as e:
        logging.error(f"Failed to retrieve processed data from MongoDB for batch {batch_id}: {e}")
        st.error(f"Database error retrieving processed CVs for batch {batch_id}: {e}")
        st.session_state.current_batch_cv_data = None # Indicate failure
        return None

    # 4. Score and Rank each CV
    with st.spinner(f"Calculating scores and ranking {len(processed_cv_data_list)} CVs..."):
        required_skills_set_lower = set(required_skills_lower)

        for cv_data in processed_cv_data_list:
            cv_filename = cv_data.get("cv_filename", "Unknown Filename")
            extracted_skills_raw = cv_data.get("extracted_skills", [])
            if not isinstance(extracted_skills_raw, list): extracted_skills_raw = []
            extracted_skills = [str(s).strip() for s in extracted_skills_raw if s and isinstance(s, (str, int, float))]

            years = cv_data.get("extracted_relevant_years", 0)
            if not isinstance(years, int):
                 try: years = int(float(years))
                 except: years = 0

            extracted_skills_lower_set = {skill.lower() for skill in extracted_skills if skill}
            matched_skills_lower = list(required_skills_set_lower.intersection(extracted_skills_lower_set))
            skill_score = sum(skill_weights.get(skill_lower, 0) for skill_lower in matched_skills_lower)

            num_required = len(required_skills_set_lower)
            num_matched = len(matched_skills_lower)
            skill_match_percent = (num_matched / num_required * 100) if num_required > 0 else 100.0

            exp_weight = assign_experience_weight(years)
            total_score = skill_score + exp_weight
            missing_skills_lower = list(required_skills_set_lower.difference(extracted_skills_lower_set))

            ranked_results.append({
                "cv_file": cv_filename,
                "total_score": round(total_score, 2),
                "years_experience": years,
                "experience_weight": exp_weight,
                "skill_score": round(skill_score, 2),
                "skill_match_percent": round(skill_match_percent, 1),
                "matched_skills_count": num_matched,
                "required_skills_count": num_required,
                "matched_skills": sorted(matched_skills_lower),
                "missing_skills": sorted(missing_skills_lower),
                "_id": cv_data.get("_id", "N/A") # Keep the string ID
            })

    ranked_results.sort(key=lambda x: (-x["total_score"], x["cv_file"]))
    logging.info(f"--- Ranking Complete for Batch: {batch_id} ---")
    logging.info(f"Ranked {len(ranked_results)} CVs.")

    for i, item in enumerate(ranked_results):
         item['rank'] = i + 1

    return ranked_results


# --- Chatbot Functionality ---
def format_cv_context_for_llm(cv_data_list):
    """ Formats the list of CV data into a concise string for the LLM prompt. """
    if not cv_data_list:
        return "No CV data available for this batch."

    context_str = "Summary of Processed CVs in Current Batch:\n"
    # Limit the number of CVs shown in context to avoid overly long prompts
    max_cvs_in_context = 20 # Adjust as needed
    limited_data = cv_data_list[:max_cvs_in_context]

    for i, cv in enumerate(limited_data):
        skills_str = ", ".join(cv.get('extracted_skills', ['N/A']))
        if len(skills_str) > 150: # Truncate long skill lists
            skills_str = skills_str[:150] + "..."
        context_str += f"- CV {i+1}: Filename: {cv.get('cv_filename', 'Unknown')}, Relevant Years: {cv.get('extracted_relevant_years', 'N/A')}, Skills: {skills_str}\n"

    if len(cv_data_list) > max_cvs_in_context:
        context_str += f"\n(Showing summary for first {max_cvs_in_context} of {len(cv_data_list)} CVs)"

    return context_str

def get_chatbot_response(user_question, cv_context_summary, chat_history):
    """ Generates a response from Gemini based on the question, CV context, and history. """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY":
        return "Error: Google API Key not configured. Chatbot is unavailable."

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # Construct conversation history for the model
    # Limit history length to avoid exceeding token limits
    history_limit = 5 # Keep last 5 pairs (user/assistant)
    model_history = []
    if chat_history:
        for msg in chat_history[-(history_limit*2):]: # Get last N messages
             role = "user" if msg["role"] == "user" else "model"
             # Ensure content is a simple string part
             model_history.append({"role": role, "parts": [msg["content"]]})


    # Create the full prompt
    prompt_parts = [
        "You are a helpful assistant analyzing CV data.",
        "You MUST answer questions based *only* on the following provided CV data summary from the current processing batch.",
        "Do not make assumptions or use external knowledge. Do not provide information beyond the scope of the CVs listed.",
        "If the answer cannot be found in the provided data, state that explicitly (e.g., 'I cannot answer that based on the provided CV summaries.').",
        "\n--- Provided CV Data Summary ---\n",
        cv_context_summary,
        "\n--- End of CV Data Summary ---\n",
        # History is passed separately via the ChatSession object if using start_chat
        # If passing directly, format history here or omit if too complex/long
        "\nUser Question:",
        user_question
    ]

    try:
        # Use start_chat for better handling of conversation history
        chat_session = model.start_chat(history=model_history)
        response = chat_session.send_message(" ".join(prompt_parts)) # Send the whole prompt content as one message

        # Log the full prompt sent (optional, for debugging)
        # logging.debug(f"Chatbot Prompt: {' '.join(prompt_parts)}")

        if not response.parts:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
            logging.error(f"Gemini Chatbot response blocked. Reason: {block_reason}")
            return f"Response blocked by safety settings (Reason: {block_reason}). Please rephrase your question."

        return response.text

    except Exception as e:
        logging.error(f"Gemini Chatbot API call error: {str(e)}", exc_info=True)
        return f"Error communicating with the chatbot API: {str(e)}"


# --- Streamlit GUI ---
st.set_page_config(layout="wide")
st.title("📊 CV Batch Processing, Ranking & Chatbot")
st.caption("Uses Gemini for analysis/chat and MongoDB for storage.")

# --- Initialize Session State ---
# For processing/ranking
if 'batch_id' not in st.session_state:
    st.session_state.batch_id = None
    st.session_state.processing_done = False
    st.session_state.ranking_done = False
    st.session_state.ranked_results = None
    st.session_state.processing_errors = []
    st.session_state.current_batch_cv_data = None # Holds data for chatbot context

# For chatbot
if "messages" not in st.session_state:
    st.session_state.messages = [] # Chat history

# --- Connect to DB ---
try:
    client, db, extracted_data_collection, job_role_collection = connect_to_mongodb_resource()
except Exception as e:
     st.error(f"Database connection failed critically: {e}. Application cannot run.")
     st.stop() # Stop if DB connection fails fundamentally

# --- Get Job Roles ---
job_roles_list = get_job_roles_data()
if not job_roles_list or "Error" in job_roles_list[0]:
     st.error(f"Failed to load job roles from database: {job_roles_list[0] if job_roles_list else 'Unknown error'}")
     job_role_options = ["Error loading roles"]
else:
     job_role_options = ["Select Job Role"] + job_roles_list


# --- GUI Layout ---
st.sidebar.header("Configuration")

# Folder Path Input
cv_folder_path = st.sidebar.text_input(
    "Enter Full Path to CV Folder:",
    help="Provide the absolute path to the folder containing PDF CVs."
)
if cv_folder_path:
    st.sidebar.caption(f"Selected folder: `{cv_folder_path}`")
    if not os.path.isdir(cv_folder_path):
         st.sidebar.warning("Path does not seem to be a valid directory.")

# Job Role Selection
selected_job_role = st.sidebar.selectbox(
    "Select Target Job Role:",
    options=job_role_options,
    index=0,
    disabled=(job_role_options[0] == "Error loading roles")
)

# API Key Status Check
api_key_ok = GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_API_KEY"
if not api_key_ok:
    st.sidebar.error("Google API Key not configured!")

# Run Button
run_disabled = (
    not cv_folder_path or
    not os.path.isdir(cv_folder_path) or
    selected_job_role == "Select Job Role" or
    selected_job_role == "Error loading roles" or
    not api_key_ok or # Disable if API key is bad
    "YOUR_MONGODB_URI" in MONGODB_URI or
    "username:password" in MONGODB_URI
)

if st.sidebar.button("▶️ Run Processing & Ranking", disabled=run_disabled, use_container_width=True):
    # --- Reset State for New Run ---
    st.session_state.processing_done = False
    st.session_state.ranking_done = False
    st.session_state.ranked_results = None
    st.session_state.batch_id = None
    st.session_state.processing_errors = []
    # --- IMPORTANT: Clear chat history and context for new batch ---
    st.session_state.messages = []
    st.session_state.current_batch_cv_data = None
    st.info("Cleared previous results and chat history. Starting new batch processing...")
    time.sleep(1) # Give user time to see the message


    # --- Start Processing ---
    st.info(f"Starting processing for job role: **{selected_job_role}** in folder: `{cv_folder_path}`")
    batch_id, success_count, error_count, error_details = process_cvs_batch(
        cv_folder_path, selected_job_role, extracted_data_collection
    )
    st.session_state.batch_id = batch_id
    st.session_state.processing_errors = error_details

    if batch_id is None:
        st.error("Batch processing could not start due to initial errors (e.g., folder access). Check logs.")
    else:
        st.session_state.processing_done = True
        if success_count > 0:
            st.success(f"Batch processing partially/fully successful (Batch ID: `{batch_id[:8]}...`). Processed: {success_count}, Failed: {error_count}.")

            # --- Start Ranking ---
            st.info("Proceeding to ranking...")
            # This function now also populates st.session_state.current_batch_cv_data
            ranked_data = rank_processed_cvs(batch_id, selected_job_role, extracted_data_collection, job_role_collection)

            if ranked_data is None:
                st.error("Ranking process failed. Chatbot context might be unavailable. Check logs.")
                # current_batch_cv_data might be None if ranking failed early
            elif not ranked_data:
                st.warning(f"Ranking complete, but no CVs met the criteria or data was insufficient for ranking (Batch ID: {batch_id[:8]}...).")
                st.session_state.ranking_done = True # Ranking ran, but yielded no results
                # current_batch_cv_data might be an empty list here
            else:
                st.session_state.ranked_results = ranked_data
                st.session_state.ranking_done = True
                st.success(f"Ranking complete! Found {len(ranked_data)} ranked CVs for batch `{batch_id[:8]}...`. Chatbot context loaded.")
        else:
            st.error(f"Batch processing completed, but no CVs were processed successfully (Batch ID: `{batch_id[:8]}...`). Cannot rank. Chatbot context unavailable.")
            # Keep processing_done=True, but ranking won't happen and context is likely None


# --- Display Area ---
st.header("Results")

if not st.session_state.processing_done and not st.session_state.ranking_done:
    st.info("Configure the CV folder path and job role in the sidebar and click 'Run Processing & Ranking'.")

# Display Processing Errors (if any)
if st.session_state.processing_errors:
    with st.expander("⚠️ Processing Errors/Warnings", expanded=(len(st.session_state.processing_errors) > 0)):
        errors_df = pd.DataFrame(st.session_state.processing_errors)
        st.dataframe(errors_df, use_container_width=True)

# Display Ranking Results
if st.session_state.ranking_done:
    if st.session_state.ranked_results:
        st.subheader(f"Ranked CVs for Batch ID: `{st.session_state.batch_id}`")
        results_df = pd.DataFrame(st.session_state.ranked_results)
        display_columns = [
            'rank', 'cv_file', 'total_score', 'skill_match_percent',
            'skill_score', 'years_experience', 'experience_weight',
            'matched_skills_count', 'required_skills_count',
            'matched_skills', 'missing_skills'
        ]
        for col in display_columns:
            if col not in results_df.columns: results_df[col] = None

        results_df['matched_skills_str'] = results_df['matched_skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        results_df['missing_skills_str'] = results_df['missing_skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

        display_df = results_df[[
            'rank', 'cv_file', 'total_score', 'skill_match_percent',
            'skill_score', 'years_experience', 'experience_weight',
            'matched_skills_count', 'required_skills_count',
            'matched_skills_str', 'missing_skills_str'
        ]].rename(columns={
            'skill_match_percent': 'Skill Match %', 'years_experience': 'Years Exp.',
            'experience_weight': 'Exp. Weight', 'skill_score': 'Skill Score',
            'total_score': 'Total Score', 'cv_file': 'CV Filename', 'rank': 'Rank',
            'matched_skills_count': '# Matched', 'required_skills_count': '# Required',
            'matched_skills_str': 'Matched Skills', 'missing_skills_str': 'Missing Skills'
        })

        st.dataframe(
             display_df, hide_index=True, use_container_width=True,
             column_config={
                 "Total Score": st.column_config.NumberColumn(format="%.2f"),
                 "Skill Score": st.column_config.NumberColumn(format="%.2f"),
                 "Skill Match %": st.column_config.NumberColumn(format="%.1f%%"),
                 "Matched Skills": st.column_config.TextColumn(width="medium"),
                 "Missing Skills": st.column_config.TextColumn(width="medium"),
             }
         )
        with st.expander("Show Full Ranked Data"):
             st.dataframe(results_df, hide_index=True)

    elif st.session_state.processing_done:
        st.info("Ranking process finished, but no ranked CVs were generated.")

# --- Chatbot Interface ---
st.divider() # Separator
st.header("💬 Chat with CV Data (Current Batch)")

if not api_key_ok:
    st.warning("Google API Key is not configured. Chatbot is disabled.")
elif st.session_state.current_batch_cv_data is None and st.session_state.processing_done:
    st.warning("CV Processing or Ranking failed to load data for the chatbot. Please check errors above or re-run.")
elif not st.session_state.processing_done:
     st.info("Process a batch of CVs first to enable the chatbot.")
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the processed CVs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare context and generate response
        with st.spinner("Thinking..."):
            cv_context_summary = format_cv_context_for_llm(st.session_state.current_batch_cv_data)
            response = get_chatbot_response(prompt, cv_context_summary, st.session_state.messages)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

