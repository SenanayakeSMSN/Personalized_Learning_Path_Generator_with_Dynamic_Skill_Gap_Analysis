"""

PUSHING THE CV DATA TO MONGODB NEW COLLECTION AND GET THE SKILLS FROM THERE
NOW WE CAN PROCESS EVEN 1000 OF CVS WITHOUT EXCEEDING THE TOKEN LIMIT

"""
# Cell 2: Import Libraries and Setup
import os
import json
import logging
import re
from datetime import datetime
from pymongo import MongoClient, errors as mongo_errors
from PIL import Image
import concurrent.futures
import uuid
import time
import traceback # For detailed error logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORTANT CONFIGURATION ---
# TODO: Verify or replace with your actual Gemini API key
GOOGLE_API_KEY = "AIzaSyBvRnSojVCuojgtGI7RisnW6-S4VpBYJWo"
# TODO: !! CRITICAL !! Verify this URI points to your target cluster "Clusterskillgapanalysislll"
MONGODB_URI = "mongodb+srv://akilapremarathna0:123@clusterskillgapanalysis.vnbcnju.mongodb.net/skillgapanalysis?retryWrites=true&w=majority"
TARGET_DB_NAME = "skillgapanalysis"
TARGET_COLLECTION_NAME = "cv_extracted_data" # Collection for extracted data
JOB_ROLE_COLLECTION_NAME = "jobrole_skill" # Collection containing job roles and required skills
# --- End Configuration ---

# --- Dependency Imports and Checks ---
try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is not installed. Run: %pip install pdfplumber")

try:
    import google.generativeai as genai
    # Configure Gemini API key globally once if possible
    if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        logging.warning("GOOGLE_API_KEY is not set. Gemini calls will fail.")
except ImportError:
    raise ImportError("google-generativeai is not installed. Run: %pip install google-generativeai")
# LangChain dependencies are removed as we use google.generativeai directly now for simplicity
# Try importing pytesseract
try:
    import pytesseract
    # Example paths (uncomment and update if needed):
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Windows
except ImportError:
    logging.warning("pytesseract not found. OCR fallback disabled. Run: %pip install pytesseract Pillow")
except Exception as e:
    logging.warning(f"Could not configure pytesseract path: {e}. Ensure Tesseract is installed and in your PATH or set the path manually.")


# --- Constants ---
MAX_WORKERS = 10 # Adjust based on system/API limits
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Or "gemini-1.5-pro" etc.

# --- Helper Functions ---
def safe_json_loads(text):
    """Safely cleans and parses JSON from text, handling markdown code blocks."""
    if not text: return None
    try:
        # Remove markdown fences and strip whitespace
        cleaned_text = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
        cleaned_text = cleaned_text.strip()
        if not cleaned_text: return None
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}. Text was: {text[:200]}...")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during JSON parsing: {e}. Text was: {text[:200]}...")
        return None


# Cell 3: Enhanced MongoDB Connection (Connects to target DB/Collection)
# def connect_to_mongodb_for_processing():
#     """Connects to MongoDB for storing/retrieving processed CV data."""
#     if "YOUR_MONGODB_URI" in MONGODB_URI or not MONGODB_URI:
#          logging.error("MONGODB_URI is not set correctly.")
#          raise ValueError("MONGODB_URI is not configured.")
#     try:
#         client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
#         client.admin.command('ping')
#         logging.info(f"Successfully connected to MongoDB cluster.")
#         db = client[TARGET_DB_NAME]
#         extracted_data_collection = db[TARGET_COLLECTION_NAME]
#         job_role_collection = db[JOB_ROLE_COLLECTION_NAME]
#         # Create indexes (idempotent)
#         extracted_data_collection.create_index("analysis_batch_id")
#         extracted_data_collection.create_index([("analysis_batch_id", 1), ("cv_filename", 1)])
#         job_role_collection.create_index("Job_Role", unique=True) # Assuming Job_Role should be unique
#         logging.info(f"Using DB '{TARGET_DB_NAME}', Collection (Extracted): '{TARGET_COLLECTION_NAME}', Collection (Job Roles): '{JOB_ROLE_COLLECTION_NAME}'")
#         return client, db, extracted_data_collection, job_role_collection
#     except mongo_errors.ConnectionFailure as e:
#         logging.error(f"MongoDB Connection Failure: {e}.")
#         raise ValueError(f"Failed to connect to MongoDB: {e}")
#     except Exception as e:
#         logging.error(f"MongoDB Setup Error: {str(e)}")
#         raise ValueError(f"Error setting up MongoDB connection: {str(e)}")

def connect_to_mongodb_for_processing():
    """Connects to MongoDB for storing/retrieving processed CV data."""
    if "YOUR_MONGODB_URI" in MONGODB_URI or not MONGODB_URI:
         logging.error("MONGODB_URI is not set correctly.")
         raise ValueError("MONGODB_URI is not configured.")
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logging.info(f"Successfully connected to MongoDB cluster.")
        db = client[TARGET_DB_NAME]
        extracted_data_collection = db[TARGET_COLLECTION_NAME]
        job_role_collection = db[JOB_ROLE_COLLECTION_NAME]
        
        # Create indexes (idempotent)
        extracted_data_collection.create_index("analysis_batch_id")
        extracted_data_collection.create_index([("analysis_batch_id", 1), ("cv_filename", 1)])
        
        # Check if there's an existing index on Job_Role
        existing_indexes = list(job_role_collection.list_indexes())
        has_job_role_index = False
        for idx in existing_indexes:
            if 'Job_Role_1' == idx.get('name'):
                has_job_role_index = True
                break
                
        # Only create the unique index if no index exists on Job_Role
        if not has_job_role_index:
            job_role_collection.create_index("Job_Role", unique=True)
        
        logging.info(f"Using DB '{TARGET_DB_NAME}', Collection (Extracted): '{TARGET_COLLECTION_NAME}', Collection (Job Roles): '{JOB_ROLE_COLLECTION_NAME}'")
        return client, db, extracted_data_collection, job_role_collection
    except mongo_errors.ConnectionFailure as e:
        logging.error(f"MongoDB Connection Failure: {e}.")
        raise ValueError(f"Failed to connect to MongoDB: {e}")
    except Exception as e:
        logging.error(f"MongoDB Setup Error: {str(e)}")
        raise ValueError(f"Error setting up MongoDB connection: {str(e)}")

# Cell 4: Enhanced CV Text Extraction (No changes needed)
def extract_cv_text(cv_path):
    """Extracts text from a PDF CV, attempting OCR as a fallback. Returns None on failure."""
    text = ""
    try:
        with pdfplumber.open(cv_path) as pdf:
            logging.debug(f"Processing PDF: {os.path.basename(cv_path)}")
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n\n"
                else:
                    # Try OCR only if pdfplumber failed and pytesseract is available
                    if 'pytesseract' in globals():
                        logging.info(f"No text from page {page_number} via pdfplumber for {os.path.basename(cv_path)}. Attempting OCR.")
                        try:
                            img = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(img, lang='eng')
                            if ocr_text and ocr_text.strip():
                                text += ocr_text + "\n\n"
                                logging.debug(f"Extracted OCR text from page {page_number} for {os.path.basename(cv_path)}")
                            else:
                                text += "\n\n" # Add space even if empty
                        except ImportError:
                            logging.error(f"OCR skipped: pytesseract/PIL not installed correctly.")
                            text += "\n\n"
                        except Exception as ocr_e:
                            if "Tesseract is not installed or" in str(ocr_e):
                                logging.error(f"OCR failed: Tesseract not found. Install/configure Tesseract.")
                            else:
                                logging.error(f"OCR failed for page {page_number} in {os.path.basename(cv_path)}: {str(ocr_e)}.")
                            text += "\n\n"
                    else:
                        logging.warning(f"OCR skipped for page {page_number} in {os.path.basename(cv_path)}: pytesseract not available.")
                        text += "\n\n" # Maintain spacing

            if not text.strip():
                logging.warning(f"No text extracted from CV '{os.path.basename(cv_path)}'.")
                return None # Signal potential issue or empty CV
            logging.debug(f"Extracted text from CV '{os.path.basename(cv_path)}'. Length: {len(text)} chars.")
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error(f"PDF '{os.path.basename(cv_path)}' is password-protected.")
        return None
    except FileNotFoundError:
        logging.error(f"CV file not found: {cv_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading CV '{os.path.basename(cv_path)}': {str(e)}")
        return None

# Cell 5: (Removed - Skills Section Extraction not used)

# Cell 6: Gemini Data Extraction (Skills & Relevant Experience - unchanged)
def extract_cv_data_with_gemini(cv_text, job_role):
    """
    Extracts skills and relevant years of experience using Gemini.
    Returns (skills, years) or (None, None) on failure.
    """
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
         logging.error("GOOGLE_API_KEY is not set. Cannot call Gemini.")
         return None, None
    if not cv_text or not cv_text.strip():
        logging.error("CV text is empty, cannot extract data.")
        return None, None
    if not job_role or not job_role.strip():
        logging.error("Job role context is empty, cannot extract relevant experience.")
        return None, None

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        current_year = datetime.now().year
        current_month = datetime.now().month
        prompt = ( # Using the same detailed prompt
            f"You are an expert in CV analysis for the job role of '{job_role}'. Extract the following from the CV text:\n\n"
            f"1. SKILLS: Extract all technical and soft skills mentioned or implied.\n"
            f"2. EXPERIENCE: Calculate the total years of professional experience relevant ONLY to the '{job_role}' role.\n\n"
            f"For SKILLS:\n"
            f"- Return distinct skills (e.g., 'PyTorch', 'AWS SageMaker').\n"
            f"- Combine variations (e.g., 'ML'/'Machine Learning' as 'Machine Learning').\n\n"
            f"For EXPERIENCE:\n"
            f"- Analyze job entries (titles, descriptions, skills) to identify those relevant to '{job_role}'.\n"
            f"- Calculate duration in months for relevant jobs: (end_year - start_year) * 12 + (end_month - start_month).\n"
            f"- Use {current_month}/{current_year} for 'Present'/'Current' end dates.\n"
            f"- Assume Jan/Dec for missing months.\n"
            f"- Sum durations (months) of relevant jobs.\n"
            f"- Convert total months to years (divide by 12), round to nearest integer.\n\n"
            f"Return a single, valid JSON object with exactly two keys:\n"
            f"- 'skills': array of unique skill strings (string[])\n"
            f"- 'years': integer representing total years of relevant experience (integer)\n\n"
            f"Example: {{\"skills\": [\"Python\", \"Machine Learning\", \"SQL\"], \"years\": 5}}\n\n"
            f"Output ONLY the JSON object, no extra text or markdown.\n\n"
            f"CV Text:\n{cv_text}"
        )

        response = model.generate_content(prompt)
        # Add safety check for response
        if not response.parts:
             logging.error(f"Gemini response blocked or empty. Reason: {response.prompt_feedback.block_reason_message if response.prompt_feedback else 'Unknown'}")
             return None, None

        raw_response = response.text.strip()
        logging.debug(f"Gemini raw response for CV data: {raw_response[:100]}...")

        cv_data = safe_json_loads(raw_response)

        if cv_data is None or not isinstance(cv_data, dict) or "skills" not in cv_data or "years" not in cv_data:
            logging.error(f"Invalid or missing JSON data from Gemini. Response: {raw_response[:200]}")
            # Basic fallback attempt
            skills_fallback = re.findall(r'"([^"]+)"', raw_response)
            years_match = re.search(r'"years":\s*(\d+)', raw_response)
            years_fallback = int(years_match.group(1)) if years_match else 0
            return skills_fallback, years_fallback

        skills = cv_data.get("skills", [])
        years = cv_data.get("years", 0)

        # Validate types
        if not isinstance(skills, list): skills = []
        skills = [str(s) for s in skills if isinstance(s, (str, int, float))]
        if not isinstance(years, int):
             try: years = int(float(years))
             except: years = 0

        logging.debug(f"Successfully extracted {len(skills)} skills, {years} relevant years for role '{job_role}'")
        return skills, years

    except Exception as e:
        logging.error(f"Gemini API call error during CV data extraction: {str(e)}", exc_info=True)
        return None, None

# Cell 7: MongoDB Query for Required Skills (Revised)
def get_required_skills_from_db(job_role, job_role_collection):
    """Fetches and cleans the list of required skills using provided collection handle."""
    try:
        # Case-insensitive search
        job_doc = job_role_collection.find_one({"Job_Role": {"$regex": f"^{re.escape(job_role)}$", "$options": "i"}})
        if not job_doc:
            logging.error(f"Job role '{job_role}' not found in collection '{JOB_ROLE_COLLECTION_NAME}'.")
            return None # Signal role not found

        required_skills_raw = job_doc.get("Required_Skills")
        if not required_skills_raw or not isinstance(required_skills_raw, str):
             logging.warning(f"'Required_Skills' missing or not a string for job role '{job_role}'.")
             return [] # Return empty list if field missing/invalid

        required_skills = [skill.strip().lower() for skill in required_skills_raw.split(",") if skill.strip()] # Normalize to lower case
        if not required_skills:
            logging.warning(f"No valid required skills derived for job role '{job_role}'. Raw: '{required_skills_raw}'")

        logging.info(f"Fetched {len(required_skills)} required skills for {job_role}.")
        return required_skills
    except Exception as e:
        logging.error(f"Error retrieving required skills for '{job_role}': {str(e)}")
        return None # Signal error

# Cell 8: Experience Weight Assignment (Unchanged)
def assign_experience_weight(years):
    """Assigns a weight based on years of relevant experience."""
    if not isinstance(years, (int, float)) or years < 0: return 0
    years = int(round(years))
    if years > 15: return 3
    elif 8 <= years <= 15: return 2
    elif 1 <= years <= 7: return 1
    else: return 0

# Cell 9: NEW - Get Skill Weights from Gemini (Only Once per Job Role)
def get_skill_weights_for_job_role(job_role, required_skills):
    """
    Uses Gemini to assign importance weights to required skills for a job role.
    Returns a dictionary {skill: weight} or None on failure.
    """
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        logging.error("GOOGLE_API_KEY not set. Cannot get skill weights.")
        return None
    if not required_skills:
        logging.warning(f"No required skills provided for '{job_role}', cannot assign weights.")
        return {} # Return empty dict

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        num_skills = len(required_skills)
        max_weight = max(5, num_skills // 2) # Dynamic max weight

        required_skills_json = json.dumps(required_skills) # Use cleaned, lowercased skills

        prompt = (
            f"You are an expert technical recruiter specializing in the '{job_role}' domain.\n\n"
            f"TASK: SKILL WEIGHTING\n"
            f"Analyze the following list of skills required for the '{job_role}' position:\n{required_skills_json}\n"
            f"Assign an integer weight to EACH required skill based on its importance and relevance specifically for the '{job_role}'. "
            f"Use a scale from 1 (least important/general) to {max_weight} (most critical technical skill). "
            f"Ensure core technical skills directly related to '{job_role}' receive higher weights.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return a single, valid JSON object where keys are the required skill strings (exactly as provided) and values are their assigned integer weights.\n\n"
            f"Example JSON Output:\n"
            f"{{\"python\": {max_weight}, \"machine learning\": {max_weight-1}, \"sql\": {max_weight // 2}, \"communication\": 1}}\n\n"
            f"IMPORTANT: Ensure the output is ONLY the JSON object, with no surrounding text, comments, or markdown formatting."
        )

        response = model.generate_content(prompt)
        if not response.parts:
             logging.error(f"Gemini response blocked or empty for skill weights. Reason: {response.prompt_feedback.block_reason_message if response.prompt_feedback else 'Unknown'}")
             return None

        raw_response = response.text.strip()
        logging.debug(f"Gemini raw skill weight response: {raw_response[:100]}...")

        weights_data = safe_json_loads(raw_response)

        if weights_data is None or not isinstance(weights_data, dict):
            logging.error(f"Invalid or missing JSON for skill weights from Gemini. Response: {raw_response[:200]}")
            return None

        # --- Validation ---
        final_weights = {}
        for skill in required_skills: # Iterate through the *expected* skills
            weight = weights_data.get(skill) # Get weight using the lowercased skill
            if not isinstance(weight, int) or weight < 1 or weight > max_weight:
                logging.warning(f"Invalid/missing weight for '{skill}' (found: {weight}). Assigning default weight 1.")
                final_weights[skill] = 1
            else:
                final_weights[skill] = weight

        # Check if any expected skills were missed entirely
        missing_from_response = set(required_skills) - set(weights_data.keys())
        if missing_from_response:
            logging.warning(f"Gemini response did not provide weights for all required skills: {missing_from_response}. Assigning default 1.")
            for skill in missing_from_response:
                 final_weights[skill] = 1 # Ensure all required skills have a weight

        logging.info(f"Successfully obtained skill weights for {len(final_weights)} skills for role '{job_role}'.")
        return final_weights

    except Exception as e:
        logging.error(f"Gemini API call error during skill weight analysis: {str(e)}", exc_info=True)
        return None

# Cell 10: Function to process a single CV (Unchanged - still focuses on extraction+storage)
def process_single_cv(cv_path, job_role, batch_id, db_collection):
    """
    Processes a single CV: Extracts text, extracts data via Gemini, stores in MongoDB.
    Returns True on success, False on failure.
    """
    cv_filename = os.path.basename(cv_path)
    logging.info(f"Processing CV: {cv_filename} for Job Context: {job_role} (Batch: {batch_id[:8]}...)")
    try:
        cv_text = extract_cv_text(cv_path)
        if cv_text is None:
            # Logged inside extract_cv_text
            return False

        extracted_skills, extracted_years = extract_cv_data_with_gemini(cv_text, job_role)
        if extracted_skills is None or extracted_years is None:
            logging.error(f"Gemini data extraction failed for {cv_filename}.")
            return False

        data_to_store = {
            "analysis_batch_id": batch_id,
            "cv_filename": cv_filename,
            "job_role_context": job_role,
            "extracted_skills": extracted_skills, # Store skills as extracted
            "extracted_relevant_years": extracted_years,
            "processed_timestamp": datetime.utcnow()
        }

        try:
            insert_result = db_collection.insert_one(data_to_store)
            logging.info(f"Stored extracted data for {cv_filename} (ID: {insert_result.inserted_id})")
            return True
        except mongo_errors.PyMongoError as e:
            logging.error(f"MongoDB insert failed for {cv_filename}: {e}")
            return False
    except Exception as e:
        logging.error(f"Critical error processing '{cv_filename}': {str(e)}", exc_info=True)
        return False

# Cell 11: Batch CV Processing Function (Unchanged - focuses on extraction+storage)
def process_cvs_batch(cv_folder, job_role, extracted_data_collection):
    """
    Processes all PDF CVs in a folder in parallel, extracts data using Gemini,
    and stores the results in MongoDB with a unique batch ID.
    Returns batch_id, success_count, error_count.
    """
    # Input validation happens in the calling GUI function now
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    logging.info(f"--- Starting CV Batch Processing ---")
    logging.info(f"Batch ID: {batch_id}")
    logging.info(f"CV Folder: {cv_folder}")
    logging.info(f"Job Role Context: {job_role}")
    logging.info(f"Max Parallel Workers: {MAX_WORKERS}")

    pdf_files = [os.path.join(cv_folder, f) for f in os.listdir(cv_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.warning(f"No PDF files found in: {cv_folder}")
        return batch_id, 0, 0

    logging.info(f"Found {len(pdf_files)} PDF files to process.")

    success_count = 0
    error_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass the MongoDB collection handle directly
        futures = {executor.submit(process_single_cv, pdf_path, job_role, batch_id, extracted_data_collection): pdf_path for pdf_path in pdf_files}

        for future in concurrent.futures.as_completed(futures):
            pdf_path = futures[future]
            cv_filename = os.path.basename(pdf_path)
            try:
                result_success = future.result()
                if result_success:
                    success_count += 1
                else:
                    error_count += 1
                    logging.warning(f"Processing flagged as failed for {cv_filename}.")
            except Exception as exc:
                logging.error(f"CV '{cv_filename}' task generated exception: {exc}", exc_info=True)
                error_count += 1

    end_time = time.time()
    logging.info(f"--- Batch Processing Complete (Batch ID: {batch_id}) ---")
    logging.info(f"Processed: {success_count}, Failed/Skipped: {error_count} in {end_time - start_time:.2f} seconds")
    return batch_id, success_count, error_count


# Cell 12: NEW - Function to Rank Processed CVs
def rank_processed_cvs(batch_id, job_role, extracted_data_collection, job_role_collection):
    """
    Retrieves processed CV data for a batch, scores, and ranks them.
    Returns a list of ranked CV dictionaries or None on failure.
    """
    logging.info(f"--- Starting Ranking for Batch: {batch_id}, Job Role: {job_role} ---")

    # 1. Fetch Required Skills
    required_skills_lower = get_required_skills_from_db(job_role, job_role_collection)
    if required_skills_lower is None:
        logging.error(f"Cannot proceed with ranking: Failed to get required skills for '{job_role}'.")
        return None # Signal failure
    if not required_skills_lower:
        logging.warning(f"No required skills found for '{job_role}'. Ranking will be based on experience only.")
        # Proceed, but skill scores will be 0

    # 2. Get Skill Weights (One Gemini Call)
    skill_weights = {}
    if required_skills_lower: # Only get weights if there are skills to weigh
        skill_weights = get_skill_weights_for_job_role(job_role, required_skills_lower)
        if skill_weights is None:
             logging.error(f"Cannot proceed with ranking: Failed to get skill weights for '{job_role}'.")
             # Optionally fallback to default weights? For now, fail.
             return None # Signal failure
        # Ensure all required skills have a weight (might be redundant if get_skill_weights handles it)
        for skill in required_skills_lower:
            if skill not in skill_weights:
                logging.warning(f"Weight missing for required skill '{skill}' after Gemini call. Assigning default 1.")
                skill_weights[skill] = 1

    # 3. Retrieve Processed Data for the Batch
    try:
        processed_cv_data = list(extracted_data_collection.find({"analysis_batch_id": batch_id}))
        if not processed_cv_data:
            logging.warning(f"No processed CV data found in MongoDB for batch_id: {batch_id}")
            return [] # Return empty list, not a failure, just no data
        logging.info(f"Retrieved {len(processed_cv_data)} processed CV records for ranking.")
    except mongo_errors.PyMongoError as e:
        logging.error(f"Failed to retrieve processed data from MongoDB for batch {batch_id}: {e}")
        return None # Signal failure

    # 4. Score and Rank
    ranked_results = []
    required_skills_set = set(required_skills_lower) # For efficient lookup

    for cv_data in processed_cv_data:
        cv_filename = cv_data.get("cv_filename", "Unknown Filename")
        extracted_skills = cv_data.get("extracted_skills", [])
        years = cv_data.get("extracted_relevant_years", 0)

        # Normalize extracted skills to lower case for comparison
        extracted_skills_lower_set = {skill.strip().lower() for skill in extracted_skills if skill.strip()}

        # Calculate matched and missing skills (locally)
        matched_skills = list(required_skills_set.intersection(extracted_skills_lower_set))
        missing_skills = list(required_skills_set.difference(extracted_skills_lower_set))
        matched_skills.sort() # For consistent display
        missing_skills.sort()

        # Calculate skill score using pre-fetched weights
        skill_score = sum(skill_weights.get(skill, 0) for skill in matched_skills) # Use .get with default 0

        # Calculate skill match percentage
        num_required = len(required_skills_set)
        num_matched = len(matched_skills)
        skill_match_percent = (num_matched / num_required * 100) if num_required > 0 else 100.0 # 100% if no skills required

        # Calculate experience weight
        exp_weight = assign_experience_weight(years)

        # --- Calculate Total Score (Example: Simple Sum, adjust weighting as needed) ---
        # total_score = (exp_weight * 0.4) + (skill_score * 0.6) # Example weighted score
        total_score = exp_weight + skill_score # Simple sum score

        ranked_results.append({
            "cv_file": cv_filename,
            "total_score": total_score,
            "years_experience": years,
            "experience_weight": exp_weight,
            "skill_score": skill_score,
            "skill_match_percent": round(skill_match_percent, 1),
            "matched_skills": matched_skills, # Lowercase
            "missing_skills": missing_skills, # Lowercase
            "required_skills": required_skills_lower, # Lowercase
            "skill_weights_used": skill_weights, # For reference
            "_id": cv_data.get("_id") # Original document ID if needed
        })

    # Sort by total score (descending)
    ranked_results.sort(key=lambda x: x["total_score"], reverse=True)

    logging.info(f"--- Ranking Complete for Batch: {batch_id} ---")
    logging.info(f"Ranked {len(ranked_results)} CVs.")
    return ranked_results


# Cell 13: Tkinter GUI (MODIFIED FOR BATCH PROCESSING AND RANKING DISPLAY)
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading

# TextHandler class remains the same
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
    def emit(self, record):
        msg = self.format(record)
        def append_message():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append_message)


def create_gui():
    """Creates and runs the Tkinter GUI for batch processing and ranking."""
    root = tk.Tk()
    root.title("CV Batch Processing & Ranking System (Gemini & MongoDB)")
    root.geometry("1100x800") # Increased size for results

    # --- Global variables / shared state ---
    all_job_roles = []
    mongo_client = None # Keep client connection open while GUI runs? Or connect/disconnect per operation? Let's try connect/disconnect for robustness.

    # --- GUI Setup ---
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1) # Configure root window resizing

    # Configure main_frame resizing
    main_frame.columnconfigure(1, weight=1) # Allow entry/combobox column to expand
    main_frame.rowconfigure(4, weight=1)    # Allow log area to expand
    main_frame.rowconfigure(5, weight=1)    # Allow results area to expand

    # --- Row 0: Folder Selection ---
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

    # --- Row 1 & 2: Job Role Search and Selection ---
    def fetch_job_roles_for_gui():
        client = None
        try:
            # TODO: Ensure MONGODB_URI points to the cluster containing JOB_ROLE_COLLECTION_NAME
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000) # Shorter timeout for GUI load
            client.admin.command('ping')
            db = client[TARGET_DB_NAME]
            collection = db[JOB_ROLE_COLLECTION_NAME]
            roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1}) if "Job_Role" in doc and doc["Job_Role"]]
            return sorted(list(set(roles)))
        except Exception as e:
            logging.error(f"Failed to fetch job roles for GUI: {str(e)}")
            messagebox.showerror("Database Error", f"Failed to fetch job roles from MongoDB: {str(e)}\nPlease check connection and '{JOB_ROLE_COLLECTION_NAME}' collection.")
            return ["Error loading roles"]
        finally:
            if client: client.close()

    all_job_roles = fetch_job_roles_for_gui()
    initial_roles_list = ["Select Job Role"] + all_job_roles if all_job_roles != ["Error loading roles"] else all_job_roles

    ttk.Label(main_frame, text="Search Job Role:").grid(row=1, column=0, padx=5, pady=(10, 0), sticky="w")
    job_search_var = tk.StringVar()
    job_search_entry = ttk.Entry(main_frame, textvariable=job_search_var, width=60)
    job_search_entry.grid(row=1, column=1, padx=5, pady=(10, 0), sticky="ew")

    ttk.Label(main_frame, text="Select Job Role:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    job_role_var = tk.StringVar()
    job_role_dropdown = ttk.Combobox(main_frame, textvariable=job_role_var, values=initial_roles_list, state="readonly", width=57)
    if initial_roles_list[0] in ["Error loading roles"]:
        job_role_var.set(initial_roles_list[0])
        job_role_dropdown.config(state=tk.DISABLED)
        job_search_entry.config(state=tk.DISABLED)
    else:
         job_role_var.set(initial_roles_list[0])
    job_role_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    def _filter_job_roles(*args): # Filter function remains the same logic
        search_term = job_search_var.get().lower().strip()
        current_selection = job_role_var.get()
        valid_roles = [r for r in all_job_roles if r not in ["Error loading roles"]]

        if not search_term:
            filtered_roles_display = ["Select Job Role"] + valid_roles
        else:
            filtered_roles = [role for role in valid_roles if search_term in role.lower()]
            filtered_roles_display = ["Select Job Role"] + filtered_roles if filtered_roles else ["No matches found"]

        job_role_dropdown['values'] = filtered_roles_display

        if current_selection in filtered_roles_display: job_role_var.set(current_selection)
        elif filtered_roles_display == ["No matches found"]:
            job_role_var.set("No matches found")
            job_role_dropdown.config(state=tk.DISABLED)
        else: job_role_var.set("Select Job Role")

        if filtered_roles_display != ["No matches found"] and initial_roles_list[0] not in ["Error loading roles"]:
             job_role_dropdown.config(state="readonly")

    job_search_var.trace_add("write", _filter_job_roles)

    # --- Row 3: Run Processing & Ranking Button ---
    def run_full_process_thread():
        # Disable controls
        run_button.config(state=tk.DISABLED)
        browse_button.config(state=tk.DISABLED)
        job_search_entry.config(state=tk.DISABLED)
        job_role_dropdown.config(state=tk.DISABLED)
        root.update_idletasks()

        cv_folder = folder_entry.get()
        job_role = job_role_var.get()

        # --- Clear previous log and results ---
        def clear_outputs():
            log_text.config(state='normal')
            log_text.delete(1.0, tk.END)
            log_text.config(state='disabled')
            results_text.config(state='normal')
            results_text.delete(1.0, tk.END)
            results_text.config(state='disabled')
        root.after(0, clear_outputs)

        # --- Input Validation ---
        if not cv_folder or not os.path.isdir(cv_folder):
            messagebox.showerror("Input Error", "Please select a valid CV folder.")
            return # Exit thread early
        if not job_role or job_role in ["Select Job Role", "No matches found", "Error loading roles"]:
            messagebox.showerror("Input Error", "Please select a valid job role.")
            return
        if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or "YOUR_MONGODB_URI" in MONGODB_URI:
             messagebox.showerror("Configuration Error", "Google API Key or MongoDB URI not set correctly.")
             return

        batch_id = None
        final_ranked_cvs = None
        mongo_client_local = None # Use local client for this thread

        try:
            # --- Connect to MongoDB ---
            # This connection will be used by both batch processing and ranking
            logging.info("Connecting to MongoDB for the operation...")
            mongo_client_local, db, extracted_data_collection, job_role_collection = connect_to_mongodb_for_processing()
            logging.info("MongoDB connection established.")

            # --- Step 1: Batch Processing ---
            batch_id, success_count, error_count = process_cvs_batch(cv_folder, job_role, extracted_data_collection)
            if success_count == 0 and error_count > 0:
                 messagebox.showerror("Processing Error", f"Batch processing failed for all CVs (Batch ID: {batch_id}). Check logs. Ranking cannot proceed.")
                 return # Exit if nothing was processed successfully
            elif success_count == 0 and error_count == 0:
                messagebox.showinfo("Processing Info", f"No PDF CVs found to process in the selected folder.")
                return # Exit if no files found

            # --- Step 2: Ranking ---
            # Only proceed if some CVs were processed and a valid batch_id exists
            if batch_id and success_count > 0:
                final_ranked_cvs = rank_processed_cvs(batch_id, job_role, extracted_data_collection, job_role_collection)
                if final_ranked_cvs is None:
                     messagebox.showerror("Ranking Error", f"Ranking failed for Batch ID: {batch_id}. Check logs for details.")
                     # Processing might have succeeded, but ranking failed
                elif not final_ranked_cvs:
                     messagebox.showinfo("Ranking Info", f"Ranking complete, but no CVs met the criteria or data was insufficient for ranking (Batch ID: {batch_id}).")
                else:
                     # --- Step 3: Display Results ---
                    def display_ranked_results():
                        results_text.config(state='normal')
                        results_text.insert(tk.END, f"--- Ranked Results for Job Role: {job_role} (Batch: {batch_id}) ---\n\n")
                        for i, cv in enumerate(final_ranked_cvs, 1):
                             results_text.insert(tk.END, f"Rank {i}: {cv['cv_file']}\n")
                             results_text.insert(tk.END, f"  Total Score: {cv['total_score']:.2f}\n") # Format score
                             results_text.insert(tk.END, f"  Relevant Experience: {cv['years_experience']} years (Weight: {cv['experience_weight']})\n")
                             results_text.insert(tk.END, f"  Skill Match: {cv['skill_match_percent']}% (Score: {cv['skill_score']:.2f})\n") # Format score
                             # Shorten lists for display
                             matched_display = cv['matched_skills']
                             missing_display = cv['missing_skills']
                             max_skills_display = 5 # Max skills to show in summary
                             if len(matched_display) > max_skills_display: matched_display = matched_display[:max_skills_display] + ['...']
                             if len(missing_display) > max_skills_display: missing_display = missing_display[:max_skills_display] + ['...']
                             results_text.insert(tk.END, f"  Matched Skills ({len(cv['matched_skills'])}): {', '.join(matched_display)}\n")
                             results_text.insert(tk.END, f"  Missing Skills ({len(cv['missing_skills'])}): {', '.join(missing_display)}\n\n")
                        results_text.config(state='disabled')
                        results_text.yview(tk.END)
                        messagebox.showinfo("Success", f"Processing and Ranking complete for Batch ID: {batch_id}.\nSee ranked results below.")
                    root.after(0, display_ranked_results)
            else:
                 messagebox.showinfo("Processing Info", f"Processing finished, but no CVs were successfully stored for ranking (Batch ID: {batch_id}).")


        except ValueError as ve: # Catch setup errors (DB connection, config)
            logging.error(f"Operation aborted due to setup error: {str(ve)}")
            messagebox.showerror("Operation Error", f"Operation could not be completed:\n{str(ve)}")
        except Exception as e:
            logging.error(f"Critical failure during operation: {str(e)}", exc_info=True)
            messagebox.showerror("Critical Error", f"An unexpected error occurred:\n{str(e)}\nPlease check the logs.")
        finally:
            # --- Close MongoDB Connection ---
            if mongo_client_local:
                mongo_client_local.close()
                logging.info("MongoDB connection closed for this operation.")
            # --- Re-enable Controls ---
            def re_enable_controls():
                run_button.config(state=tk.NORMAL)
                browse_button.config(state=tk.NORMAL)
                job_search_entry.config(state=tk.NORMAL)
                if initial_roles_list[0] not in ["Error loading roles"]:
                     _filter_job_roles() # Reset dropdown state correctly
            root.after(0, re_enable_controls)

    # Function to start the processing & ranking thread
    def start_full_process():
        if messagebox.askyesno("Confirm Operation", "Start processing AND ranking all PDF CVs in the selected folder?\nThis may take time and incur API costs."):
            full_process_thread = threading.Thread(target=run_full_process_thread, daemon=True)
            full_process_thread.start()

    run_button = ttk.Button(main_frame, text="Run Processing & Ranking", command=start_full_process)
    run_button.grid(row=2, column=2, padx=5, pady=5, sticky="w")

    # --- Row 4: Log Area ---
    log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
    log_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log_text = scrolledtext.ScrolledText(log_frame, width=120, height=15, state='disabled', wrap=tk.WORD) # Adjusted height
    log_text.grid(row=0, column=0, sticky="nsew")

    # Setup logging handler
    text_handler = TextHandler(log_text)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    text_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(text_handler)
    # Check if root logger already has handlers to avoid duplicates if script is re-run in some environments
    if not logging.getLogger().handlers:
        logging.getLogger().addHandler(text_handler)
    logging.getLogger().setLevel(logging.INFO)


    # --- Row 5: Results Area ---
    results_frame = ttk.LabelFrame(main_frame, text="Ranked CV Results", padding="5")
    results_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
    results_frame.columnconfigure(0, weight=1)
    results_frame.rowconfigure(0, weight=1)
    results_text = scrolledtext.ScrolledText(results_frame, width=120, height=15, state='disabled', wrap=tk.WORD) # Adjusted height
    results_text.grid(row=0, column=0, sticky="nsew")


    # Start the Tkinter event loop
    logging.info("Batch Processing & Ranking GUI Ready.")
    root.mainloop()

# Cell 14: Run the application
if __name__ == "__main__":
    # Basic Pre-checks
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: GOOGLE_API_KEY is not set correctly.")
        logging.critical("GOOGLE_API_KEY is not set correctly.")
    if "YOUR_MONGODB_URI" in MONGODB_URI:
        print(f"ERROR: MONGODB_URI is not set correctly. Update for target cluster.")
        logging.critical(f"MONGODB_URI is not set correctly.")

    # Start GUI
    create_gui()