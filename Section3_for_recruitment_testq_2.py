"""

IN THIS CODE CV DATA WONT BE PUSHIN TO THE MONGODB
JUST KEEP IT IN THE RAM AND PROCESS EVERYTHING
ADDED A SERACH BAR FOR THE JOB ROLE



"""
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
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Example
except ImportError:
    raise ImportError("pytesseract or PIL is not installed. Run: %pip install pytesseract Pillow")
except Exception as e:
    # Catch potential errors if Tesseract is not found even if pytesseract is installed
    logging.warning(f"Could not configure pytesseract path: {e}. Ensure Tesseract is installed and in your PATH or set the path manually.")


from langchain_core.messages import HumanMessage, SystemMessage

# Cell 3: MongoDB Connection
def connect_to_mongodb():
    """Connects to MongoDB and fetches the list of job roles."""
    if MONGODB_URI == "YOUR_MONGODB_URI":
         logging.error("MONGODB_URI is not set. Please replace 'YOUR_MONGODB_URI' or set the environment variable.")
         raise ValueError("MONGODB_URI is not configured.")
    try:
        client = MongoClient(MONGODB_URI)
        # Ping the server to confirm connection
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")
        db = client["skillgapanalysis"] # Use your actual database name
        collection = db["jobrole_skill"] # Use your actual collection name
        # Ensure index exists for faster lookups
        collection.create_index("Job_Role")
        # Fetch roles, handling potential missing fields or empty collection
        job_roles = [doc["Job_Role"] for doc in collection.find({}, {"Job_Role": 1}) if "Job_Role" in doc and doc["Job_Role"]]
        job_roles = sorted(list(set(job_roles))) # Get unique, sorted roles
        if not job_roles:
            logging.warning("No job roles found in the database collection 'jobrole_skill'. Check the database and collection name.")
        logging.info(f"Connected to MongoDB. Found {len(job_roles)} unique job roles.")
        return collection, job_roles
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB or fetch job roles: {str(e)}")
        # Raise a more specific error or return defaults depending on desired behavior
        raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

# Cell 4: Enhanced CV Text Extraction
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
                        else:
                            logging.error(f"OCR failed for page {page_number}: {str(ocr_e)}. Check Tesseract installation/path.")
                        text += "\n\n" # Add space to maintain structure
            if not text.strip():
                logging.error(f"No text extracted from CV '{os.path.basename(cv_path)}' after attempting both standard extraction and OCR.")
                raise ValueError("No text could be extracted from the CV PDF. Check the file content/format or Tesseract setup if OCR was needed.")
            logging.info(f"Successfully extracted text from CV '{os.path.basename(cv_path)}'. Length: {len(text)} chars.")
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error(f"PDF file '{os.path.basename(cv_path)}' is password-protected. Cannot process.")
        raise ValueError(f"'{os.path.basename(cv_path)}' is password-protected.")
    except FileNotFoundError:
        logging.error(f"CV file not found at path: {cv_path}")
        raise ValueError(f"CV file not found: {cv_path}")
    except Exception as e:
        logging.error(f"Error reading CV '{os.path.basename(cv_path)}': {str(e)}")
        # Raise a more specific error or return None/empty string depending on desired behavior
        raise ValueError(f"Error processing CV '{os.path.basename(cv_path)}': {str(e)}")


# Cell 5: Skills Section Extraction (Optional Enhancement - Not used by Gemini in this version)
# Note: The current Gemini prompt analyzes the *entire* CV text for skills, making this function less critical
# unless you want to pre-filter or focus the analysis later.
def extract_skills_section(cv_text):
    """Attempts to extract a dedicated skills section using regex."""
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
        logging.warning("No explicit skills section found using regex patterns. Using full CV text for analysis.")
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
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
         logging.error("GOOGLE_API_KEY is not set. Cannot call Gemini.")
         raise ValueError("GOOGLE_API_KEY is not configured.")
    if not cv_text or not cv_text.strip():
        logging.error("CV text is empty, cannot extract data.")
        return [], 0
    if not job_role or not job_role.strip():
        logging.error("Job role is empty, cannot calculate relevant experience.")
        # Return empty skills but maybe allow analysis without role-specific experience?
        # For now, returning empty for both as role is key to relevance.
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
            # Ensure skills are strings
            skills = [str(s) for s in skills if isinstance(s, (str, int, float))] # Convert non-strings if possible

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
            logging.error(f"Failed to parse Gemini JSON response for skills/years. Error: {str(e)}. Response attempted: {cleaned_response[:200]}")
            # Attempt basic regex fallback if JSON parsing fails
            skills_fallback = re.findall(r'"([^"]+)"', cleaned_response) # Look for quoted strings as skills
            years_match = re.search(r'"years":\s*(\d+)', cleaned_response) # Look for "years": number pattern
            years_fallback = int(years_match.group(1)) if years_match else 0
            logging.warning(f"Using regex fallback: Found {len(skills_fallback)} potential skills and {years_fallback} years.")
            return skills_fallback, years_fallback
        except ValueError as ve:
            logging.error(f"Data validation error after parsing skills/years: {ve}. Response: {cleaned_response}")
            return [], 0 # Return empty/zero on validation error

    except Exception as e:
        # Catch potential API errors, configuration errors, etc.
        logging.error(f"Error during Gemini API call for CV data extraction: {str(e)}")
        # Consider more specific error handling based on google.generativeai exceptions if needed
        return [], 0 # Return empty list and 0 years on error

# Cell 7: MongoDB Query for Required Skills
def get_required_skills(job_collection, job_role):
    """Fetches and cleans the list of required skills for a given job role."""
    try:
        # Case-insensitive search for job role might be more robust
        # Use regex for exact match, case-insensitive
        job_doc = job_collection.find_one({"Job_Role": {"$regex": f"^{re.escape(job_role)}$", "$options": "i"}})
        if not job_doc:
            logging.error(f"Job role '{job_role}' not found in database (case-insensitive search).")
            # Returning empty list instead of raising error, allows GUI to proceed maybe?
            # Or re-raise to enforce valid role selection. Let's re-raise for clarity.
            raise ValueError(f"Job role '{job_role}' not found in database")

        required_skills_raw = job_doc.get("Required_Skills")
        if not required_skills_raw or not isinstance(required_skills_raw, str):
             logging.warning(f"Required_Skills field missing or not a string for job role '{job_role}'. No required skills loaded.")
             # Raise error or return empty list? Returning empty list might be safer for processing flow.
             return []
             # raise ValueError(f"Required skills format error for job role '{job_role}'")

        # Clean up skills: split by comma, strip whitespace, remove empty strings
        required_skills = [skill.strip() for skill in required_skills_raw.split(",") if skill.strip()]
        if not required_skills:
            logging.warning(f"No required skills derived after cleaning for job role '{job_role}'. Raw value was: '{required_skills_raw}'")
            return [] # Return empty list if no valid skills found

        logging.info(f"Required skills for {job_role}: {required_skills}")
        return required_skills
    except Exception as e:
        logging.error(f"Error retrieving required skills for '{job_role}': {str(e)}")
        # Re-raise to indicate failure in the calling function
        raise ValueError(f"Error retrieving required skills for '{job_role}': {str(e)}")

# Cell 8: Experience Weight Assignment
def assign_experience_weight(years):
    """Assigns a weight based on years of relevant experience."""
    if not isinstance(years, (int, float)) or years < 0:
        logging.warning(f"Invalid years value '{years}' received for weight assignment. Assigning weight 0.")
        return 0
    years = int(round(years)) # Round to nearest integer for comparison

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
        weights = {skill: 1 for skill in required_skills} # Assign default weight 1
        return weights, required_skills # All required skills are missing

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        num_skills = len(required_skills)
        max_weight = max(5, num_skills // 2) # Base max weight on number of skills, minimum 5

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
                 logging.warning(f"Expected 'weights' to be a dict, got {type(weights)}. Using fallback weights.")
                 weights = {skill: 1 for skill in required_skills} # Fallback weights

            if not isinstance(missing_skills, list):
                logging.warning(f"Expected 'missing_skills' to be a list, got {type(missing_skills)}. Using fallback.")
                # Attempt recovery or return all as missing
                missing_skills = list(missing_skills) if isinstance(missing_skills, (tuple, set)) else required_skills

            # Ensure all required skills have a weight (assign default if missing) and correct type
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


            # Ensure missing skills are actually from the required list (case-sensitive check against input list)
            validated_missing = [ms for ms in missing_skills if ms in required_skills]
            if len(validated_missing) != len(missing_skills):
                missed_original = set(missing_skills)
                missed_validated = set(validated_missing)
                extra_reported = missed_original - missed_validated
                logging.warning(f"Gemini reported missing skills not in the original required list: {extra_reported}. Filtering them out.")
                missing_skills = validated_missing
            # --- End Validation ---

            logging.info(f"Analyzed weights for {len(final_weights)} skills and identified {len(missing_skills)} missing skills for role '{job_role}'")
            return final_weights, missing_skills
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
    if not job_role or job_role in ["No roles found", "Error loading roles", "No matches found", "Select Job Role"]:
        logging.error(f"Invalid job role '{job_role}' selected for ranking.")
        raise ValueError("A valid job role must be selected for analysis.")

    job_collection = None
    required_skills = []
    try:
        job_collection, _ = connect_to_mongodb() # Connect but ignore roles list here
        required_skills = get_required_skills(job_collection, job_role)
        if not required_skills:
             # Log warning instead of error if function returns empty list by design
             logging.warning(f"No required skills found or defined for job role '{job_role}'. Ranking may be based only on experience.")
             # Proceeding without required skills is possible, but skill score will be 0.
             # Let's allow this, but the results might not be meaningful.

    except ValueError as ve: # Catch errors from connect_to_mongodb or get_required_skills
        logging.error(f"Setup error for ranking: {str(ve)}")
        # Re-raise to stop the ranking process if DB connection or role lookup fails fundamentally
        raise ve

    cv_results = []
    cv_texts = {}  # Store CV text for chatbot

    logging.info(f"Starting CV processing in folder: {cv_folder} for job role: {job_role}")
    processed_files = 0
    error_files = 0

    # --- Pre-analyze weights and required skills once if possible ---
    # Note: This is tricky because missing skills depend on *each* CV.
    # We can only pre-calculate weights if they are solely job-role dependent.
    # The current combined analyze_job_requirements needs CV skills, so we call it per CV.
    # If weights were static per role, we could fetch them once here.

    pdf_files_found = [f for f in os.listdir(cv_folder) if f.lower().endswith('.pdf')]
    if not pdf_files_found:
        logging.warning(f"No PDF files found in the specified folder: {cv_folder}")
        return [], {} # Return empty if no PDFs

    logging.info(f"Found {len(pdf_files_found)} PDF files to process.")

    for cv_file in pdf_files_found:
        cv_path = os.path.join(cv_folder, cv_file)
        logging.info(f"--- Processing CV file: {cv_file} ---")
        try:
            # 1. Extract text (handles potential errors inside)
            cv_text = extract_cv_text(cv_path)
            cv_texts[cv_file] = cv_text  # Store CV text for potential chatbot use

            # 2. Call Gemini to extract skills and *relevant* experience years
            # Pass the full CV text and the specific job role
            cv_skills, years = extract_cv_data_with_gemini(cv_text, job_role) # MODIFIED CALL

            if not cv_skills and years == 0:
                # Check if this was due to an API error or genuinely no data found
                # Logging inside extract_cv_data_with_gemini should indicate API errors
                logging.warning(f"No skills or relevant experience years extracted from '{cv_file}' by Gemini. Ranking scores may be low.")
                # Continue processing, score will reflect lack of data.

            # 3. Assign experience weight based on relevant years
            exp_weight = assign_experience_weight(years)

            # 4. Call Gemini to get skill weights and identify missing skills *for this CV*
            # Handle the case where required_skills might be empty
            skill_weights = {}
            missing_skills = list(required_skills) # If no required skills, this is empty. If no CV skills, all required are missing.
            skill_score = 0
            matched_skills_list = []

            if required_skills: # Only analyze/score skills if required skills are defined
                skill_weights, missing_skills = analyze_job_requirements(job_role, required_skills, cv_skills)

                # 5. Calculate skill score based on weights of *matched* skills
                # Matched skills are required skills NOT in the missing list
                matched_skills_set = set(required_skills) - set(missing_skills)
                matched_skills_list = sorted(list(matched_skills_set)) # For display

                skill_score = sum(skill_weights.get(skill, 0) for skill in matched_skills_list)
            else:
                # If no required skills, skill score is 0, missing skills is empty, matched is empty.
                logging.info(f"No required skills defined for '{job_role}', skill score set to 0 for '{cv_file}'.")


            # 6. Calculate skill match percentage
            num_required = len(required_skills)
            num_matched = len(matched_skills_list)
            skill_match_percent = (num_matched / num_required * 100) if num_required > 0 else 100 if not required_skills else 0
            # Edge case: If no required skills, is it 100% match? Or 0%? Let's say 100% as there are no unmet requirements.

            # 7. Calculate total score (simple sum, can be weighted)
            # Example weighted: total_score = (exp_weight * 0.4) + (skill_score * 0.6)
            total_score = exp_weight + skill_score

            cv_results.append({
                "cv_file": cv_file,
                "years_experience": years,          # Relevant years identified by Gemini
                "experience_weight": exp_weight,
                "cv_skills": cv_skills,             # Skills found in CV by Gemini
                "required_skills": required_skills, # Skills needed for the job from DB
                "matched_skills": matched_skills_list, # Required skills found in CV
                "missing_skills": missing_skills,   # Required skills not found in CV
                "skill_score": skill_score,         # Weighted score of matched skills
                "skill_weights": skill_weights,     # Weights assigned to each required skill
                "skill_match_percent": round(skill_match_percent, 1), # Percentage of required skills matched
                "total_score": total_score          # Combined score
            })
            logging.info(f"Successfully processed '{cv_file}': Score = {total_score}, Relevant Years = {years}, Match = {round(skill_match_percent, 1)}%")
            processed_files += 1

        except ValueError as ve: # Catch specific errors from text extraction, API key issues, role not found etc.
            logging.error(f"Skipping '{cv_file}' due to processing error: {str(ve)}")
            error_files += 1
            # Store minimal info for error reporting if needed
            cv_texts.pop(cv_file, None) # Remove text if processing failed badly
            continue # Skip to next file
        except Exception as e:
            logging.error(f"Unexpected critical error processing '{cv_file}': {str(e)}", exc_info=True) # Log stack trace
            error_files += 1
            cv_texts.pop(cv_file, None)
            continue # Skip to next file

    # Sort CVs by total score in descending order
    ranked_cvs = sorted(cv_results, key=lambda x: x["total_score"], reverse=True)

    logging.info(f"--- CV Ranking Complete for role '{job_role}' ---")
    logging.info(f"Successfully processed: {processed_files} CVs")
    logging.info(f"Skipped due to errors: {error_files} CVs")
    logging.info(f"Total PDF files found: {len(pdf_files_found)}")
    logging.info(f"Total CVs ranked: {len(ranked_cvs)}")
    if not ranked_cvs and pdf_files_found:
        logging.warning("No CVs were successfully ranked, though PDF files were found. Check logs for errors.")
    elif not ranked_cvs and not pdf_files_found:
         logging.warning("No PDF CVs found in the folder to rank.")


    return ranked_cvs, cv_texts # Return ranked list and the dictionary of CV texts

# Cell 11: Tkinter GUI with Chatbot (MODIFIED FOR JOB ROLE SEARCH)
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading # For running analysis in background

# Handler to redirect logging to the Tkinter text widget
class TextHandler(logging.Handler):
    """A logging handler that redirects logs to a Tkinter ScrolledText widget."""
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
        # Optional: Add a lock for thread safety if multiple threads log
        # self.lock = threading.Lock()

    def emit(self, record):
        msg = self.format(record)
        def append_message():
            # with self.lock: # Use lock if using threads for logging
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END) # Auto-scroll
        # Schedule the GUI update in the main Tkinter thread
        self.text_widget.after(0, append_message)

def create_gui():
    """Creates and runs the Tkinter GUI."""
    root = tk.Tk()
    root.title("CV Ranking System with Chatbot (Gemini & MongoDB)")
    root.geometry("1100x800") # Slightly larger window for search bar

    # --- Global variables for GUI state ---
    cv_texts = {}       # Dictionary to store CV text: {filename: text}
    ranked_cvs = []     # List to store analysis results: [ {cv_data_dict}, ... ]
    all_job_roles = []  # FULL list of job roles from DB
    llm = None          # Placeholder for the LangChain LLM

    # --- Initialize LangChain Gemini model ---
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        logging.error("FATAL: GOOGLE_API_KEY is not set. Chatbot and analysis will not function.")
        messagebox.showerror("API Key Error", "GOOGLE_API_KEY is not configured.\nPlease set it in the script or environment variable.\nChatbot functionality will be disabled.")
        llm = None
    else:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", # Or "gemini-pro"
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2, # Lower temperature for more factual answers
                # Adjust safety settings if needed, e.g.:
                # safety_settings = {
                #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                # }
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
        if not cv_filename or cv_filename in ["Select CV after analysis", "No CVs ranked"] or cv_filename not in cv_texts:
            return "Please select a valid CV from the dropdown first (after running analysis)."
        if not question:
            return "Please enter a question."

        try:
            job_role = job_role_var.get() # Get the currently selected job role
            cv_text = cv_texts.get(cv_filename, "CV text not found for the selected file.")
            # Find the analysis data for the selected CV
            selected_cv_analysis = next((item for item in ranked_cvs if item["cv_file"] == cv_filename), None)

            # Prepare analysis summary (avoid overly long context if possible)
            if selected_cv_analysis:
                analysis_summary_dict = {
                    "cv_file": selected_cv_analysis.get("cv_file"),
                    "total_score": selected_cv_analysis.get("total_score"),
                    "years_experience": selected_cv_analysis.get("years_experience"),
                    "skill_match_percent": selected_cv_analysis.get("skill_match_percent"),
                    "matched_skills": selected_cv_analysis.get("matched_skills"),
                    "missing_skills": selected_cv_analysis.get("missing_skills"),
                }
                analysis_summary = json.dumps(analysis_summary_dict, indent=2)
            else:
                analysis_summary = "Analysis data not found for this CV."


            # --- Answering Prompt ---
            # Provide context: job role, specific CV text, and its analysis results
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
            return f"Sorry, an error occurred while processing your question: {str(e)}"

    # --- GUI Setup ---
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

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

    # --- Row 1 & 2: Job Role Search and Selection (MODIFIED) ---
    # Fetch job roles at startup
    try:
        _, fetched_roles = connect_to_mongodb()
        all_job_roles = fetched_roles # Keep the full list
        if not all_job_roles:
            all_job_roles = ["No roles found in DB"] # Placeholder
            messagebox.showwarning("Database Info", "No job roles found in the database. Please check the MongoDB connection and the collection 'jobrole_skill'. Analysis requires a valid role.")
            initial_roles_list = all_job_roles
        else:
            initial_roles_list = ["Select Job Role"] + all_job_roles # Add a prompt
    except Exception as e:
        all_job_roles = ["Error loading roles"]
        initial_roles_list = all_job_roles
        messagebox.showerror("Database Error", f"Failed to fetch job roles from MongoDB: {str(e)}\nAnalysis cannot proceed without job roles.")
        logging.error(f"Failed to fetch job roles: {str(e)}")

    # Row 1: Search Label and Entry
    ttk.Label(main_frame, text="Search Job Role:").grid(row=1, column=0, padx=5, pady=(10, 0), sticky="w")
    job_search_var = tk.StringVar()
    job_search_entry = ttk.Entry(main_frame, textvariable=job_search_var, width=60)
    job_search_entry.grid(row=1, column=1, padx=5, pady=(10, 0), sticky="ew")

    # Row 2: Selection Label and Combobox
    ttk.Label(main_frame, text="Select Job Role:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
    job_role_var = tk.StringVar()
    job_role_dropdown = ttk.Combobox(main_frame, textvariable=job_role_var, values=initial_roles_list, state="readonly", width=57)

    # Set initial state and value
    if initial_roles_list[0] in ["No roles found in DB", "Error loading roles"]:
        job_role_var.set(initial_roles_list[0])
        job_role_dropdown.config(state=tk.DISABLED)
        job_search_entry.config(state=tk.DISABLED)
    else:
         job_role_var.set(initial_roles_list[0]) # Set to "Select Job Role" initially

    job_role_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    # --- Filter Function (callback for search entry changes) ---
    def _filter_job_roles(*args):
        search_term = job_search_var.get().lower().strip()
        current_selection = job_role_var.get()

        # Filter the *actual* roles (excluding placeholders/errors)
        valid_roles = [r for r in all_job_roles if r not in ["No roles found in DB", "Error loading roles"]]

        if not search_term: # If search is empty, show all valid roles + prompt
            filtered_roles_display = ["Select Job Role"] + valid_roles
        else:
            # Filter based on search term (case-insensitive)
            filtered_roles = [role for role in valid_roles if search_term in role.lower()]
            if filtered_roles:
                filtered_roles_display = ["Select Job Role"] + filtered_roles # Keep prompt if matches found
            else:
                 filtered_roles_display = ["No matches found"] # Special placeholder

        # Update dropdown values
        job_role_dropdown['values'] = filtered_roles_display

        # Decide what to do with the current selection
        if current_selection in filtered_roles_display:
             # If the old selection is still valid in the new list, keep it
            job_role_var.set(current_selection)
        elif filtered_roles_display == ["No matches found"]:
             # If no matches, set to the placeholder
            job_role_var.set("No matches found")
            job_role_dropdown.config(state=tk.DISABLED) # Disable selection
        else:
            # Otherwise (e.g., search cleared, old selection invalid), reset to prompt
            job_role_var.set("Select Job Role")

        # Ensure dropdown is enabled if there are selectable roles
        if filtered_roles_display != ["No matches found"] and initial_roles_list[0] not in ["No roles found in DB", "Error loading roles"]:
             job_role_dropdown.config(state="readonly")

    # Bind the filter function to changes in the search variable
    job_search_var.trace_add("write", _filter_job_roles)


    # --- Row 3: Run Analysis Button ---
    # This function runs in a separate thread
    def run_analysis_thread():
         # Disable button/inputs during analysis
        run_button.config(state=tk.DISABLED)
        browse_button.config(state=tk.DISABLED)
        job_search_entry.config(state=tk.DISABLED)
        job_role_dropdown.config(state=tk.DISABLED)
        root.update_idletasks() # Ensure GUI updates

        nonlocal cv_texts, ranked_cvs # Modify shared state
        cv_folder = folder_entry.get()
        job_role = job_role_var.get()

        # --- Clear previous results and log ---
        # Need to schedule GUI updates from this thread via `after`
        def clear_results():
            result_text.config(state='normal')
            result_text.delete(1.0, tk.END)
            result_text.config(state='disabled')
            chat_text.config(state='normal')
            chat_text.delete(1.0, tk.END)
            chat_text.insert(tk.END, "Chat history cleared.\n") # Indicate clearing
            chat_text.config(state='disabled')
            # Clear CV dropdown for chatbot
            cv_dropdown['values'] = ["Select CV after analysis"]
            cv_var.set("Select CV after analysis")
            cv_dropdown.config(state=tk.DISABLED)
        root.after(0, clear_results)

        # --- Input Validation ---
        if not cv_folder or not os.path.isdir(cv_folder):
            logging.error("Invalid CV folder selected.")
            messagebox.showerror("Input Error", "Please select a valid folder containing CVs.")
            # Re-enable widgets in finally block
            return
        if not job_role or job_role in ["Select Job Role", "No roles found in DB", "Error loading roles", "No matches found"]:
            logging.error(f"Invalid job role selected: '{job_role}'")
            messagebox.showerror("Input Error", "Please select a valid job role from the dropdown.")
            # Re-enable widgets in finally block
            return
        if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or MONGODB_URI == "YOUR_MONGODB_URI":
             logging.error("API Key or MongoDB URI not configured.")
             messagebox.showerror("Configuration Error", "Google API Key or MongoDB URI is not set correctly in the script or environment variables. Analysis cannot proceed.")
             return

        analysis_success = False
        try:
            logging.info(f"--- Starting Analysis for Job Role: {job_role} in Folder: {cv_folder} ---")

            # Run the core ranking function
            ranked_cvs, cv_texts = rank_cvs(cv_folder, job_role)
            analysis_success = True # Mark success if rank_cvs completes without error

            # --- Update Results Text Area (via `after`) ---
            def update_results_display():
                result_text.config(state='normal')
                result_text.insert(tk.END, f"--- Analysis Results for Job Role: {job_role} ---\n\n")
                if ranked_cvs:
                    result_text.insert(tk.END, f"Ranked {len(ranked_cvs)} CVs:\n")
                    for i, cv in enumerate(ranked_cvs, 1):
                        result_text.insert(tk.END, f"\nRank {i}: {cv['cv_file']}\n")
                        result_text.insert(tk.END, f"  Total Score: {cv['total_score']}\n")
                        result_text.insert(tk.END, f"  Relevant Experience: {cv['years_experience']} years (Weight: {cv['experience_weight']})\n")
                        result_text.insert(tk.END, f"  Skill Match: {cv['skill_match_percent']}% (Score: {cv['skill_score']})\n")
                        # Shorten long lists for display
                        matched_display = cv['matched_skills']
                        missing_display = cv['missing_skills']
                        if len(matched_display) > 7: matched_display = matched_display[:7] + ['...']
                        if len(missing_display) > 7: missing_display = missing_display[:7] + ['...']
                        result_text.insert(tk.END, f"  Matched Skills ({len(cv['matched_skills'])}): {', '.join(matched_display)}\n")
                        result_text.insert(tk.END, f"  Missing Skills ({len(cv['missing_skills'])}): {', '.join(missing_display)}\n")
                        # Optionally show CV skills:
                        # cv_skills_display = cv['cv_skills']
                        # if len(cv_skills_display) > 10: cv_skills_display = cv_skills_display[:10] + ['...']
                        # result_text.insert(tk.END, f"  CV Skills ({len(cv['cv_skills'])}): {', '.join(cv_skills_display)}\n")

                else:
                    result_text.insert(tk.END, "No CVs were successfully processed or ranked.\nPlease check the logs above and ensure PDF files exist in the selected folder and the job role is correctly defined in the database.\n")
                result_text.config(state='disabled')
                result_text.yview(tk.END) # Scroll to end

                # --- Update CV Dropdown for Chatbot ---
                cv_files = [cv['cv_file'] for cv in ranked_cvs] if ranked_cvs else ["No CVs ranked"]
                cv_dropdown['values'] = cv_files
                if cv_files[0] != "No CVs ranked":
                    cv_var.set(cv_files[0]) # Select the top-ranked CV
                    cv_dropdown.config(state="readonly")
                else:
                    cv_var.set("No CVs ranked")
                    cv_dropdown.config(state=tk.DISABLED)

            root.after(0, update_results_display)

        except ValueError as ve: # Catch errors raised by rank_cvs (e.g., bad folder, role not found)
            logging.error(f"Analysis aborted due to validation error: {str(ve)}")
            messagebox.showerror("Analysis Error", f"Analysis could not be completed:\n{str(ve)}")
            # Results area remains empty or shows previous results until cleared next time
        except Exception as e:
            logging.error(f"Analysis failed critically during execution: {str(e)}", exc_info=True)
            messagebox.showerror("Analysis Error", f"An unexpected error occurred during analysis:\n{str(e)}\nPlease check the logs.")
            # Results area remains empty or shows previous results
        finally:
             # --- Re-enable Controls (via `after`) ---
            def re_enable_controls():
                run_button.config(state=tk.NORMAL)
                browse_button.config(state=tk.NORMAL)
                job_search_entry.config(state=tk.NORMAL)
                # Only re-enable dropdown if it wasn't disabled due to load error
                if initial_roles_list[0] not in ["No roles found in DB", "Error loading roles"]:
                    # Re-apply filter in case search term exists
                     _filter_job_roles() # This will set the correct state (readonly/disabled)
                # Re-enable CV dropdown ONLY if analysis was successful *and* found CVs
                if analysis_success and ranked_cvs:
                    cv_dropdown.config(state="readonly")
                else:
                     cv_dropdown.config(state=tk.DISABLED)


            root.after(0, re_enable_controls)

    # Function to start the analysis thread
    def start_analysis():
        # Create and start the thread
        analysis_thread = threading.Thread(target=run_analysis_thread, daemon=True)
        analysis_thread.start()

    run_button = ttk.Button(main_frame, text="Run Analysis", command=start_analysis)
    # Place button next to the job role dropdown
    run_button.grid(row=2, column=2, padx=5, pady=5, sticky="w")

    # --- Row 4: Results Area (Log and Ranking Output) ---
    # Increased row number due to added search row
    results_frame = ttk.LabelFrame(main_frame, text="Analysis Log & Results", padding="5")
    results_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
    results_frame.columnconfigure(0, weight=1)
    results_frame.rowconfigure(0, weight=1) # Make scrolled text expand

    result_text = scrolledtext.ScrolledText(results_frame, width=120, height=18, state='disabled', wrap=tk.WORD)
    result_text.grid(row=0, column=0, sticky="nsew")

    # Setup logging handler to redirect to the results text area
    text_handler = TextHandler(result_text)
    # Configure format for the handler
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    text_handler.setFormatter(log_formatter)
    # Add the handler to the root logger
    logging.getLogger().addHandler(text_handler)
    # Optionally set the level for the handler (e.g., INFO or DEBUG)
    # text_handler.setLevel(logging.INFO) # Not usually needed, root logger level controls it
    logging.getLogger().setLevel(logging.INFO) # Ensure INFO level messages are captured


    # --- Row 5: Chatbot Area ---
    # Increased row number
    chatbot_frame = ttk.LabelFrame(main_frame, text="Chatbot - Ask about a specific CV", padding="5")
    chatbot_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
    chatbot_frame.columnconfigure(1, weight=1) # Allow question entry to expand
    # Configure row weights for chatbot elements
    chatbot_frame.rowconfigure(1, weight=1) # Chat history area expands

    # Row 5, Sub-row 0: CV Selection for Chatbot
    ttk.Label(chatbot_frame, text="Select CV:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    cv_var = tk.StringVar()
    cv_dropdown = ttk.Combobox(chatbot_frame, textvariable=cv_var, values=["Select CV after analysis"], state=tk.DISABLED, width=40)
    cv_dropdown.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="w") # Adjusted padding

    # Row 5, Sub-row 1: Chat History
    chat_text = scrolledtext.ScrolledText(chatbot_frame, width=100, height=10, state='disabled', wrap=tk.WORD)
    chat_text.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

    # Row 5, Sub-row 2: Question Entry & Send Button
    question_entry = ttk.Entry(chatbot_frame, width=80)
    question_entry.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    question_entry.bind("<Return>", lambda event: send_question()) # Bind Enter key

    # Function to handle sending a question (runs in main thread for simplicity, but LLM call can block)
    def send_question():
        nonlocal llm # Access the LLM instance
        question = question_entry.get().strip()
        selected_cv = cv_var.get()

        if not llm:
             messagebox.showerror("Chatbot Error", "Chatbot is not initialized. Cannot send question.")
             return

        if selected_cv in ["Select CV after analysis", "No CVs ranked"] or not selected_cv:
             messagebox.showwarning("Chatbot", "Please run the analysis and select a valid CV from the dropdown first.")
             return
        if not question:
            messagebox.showwarning("Chatbot", "Please enter a question.")
            return

        # --- Add question to chat history immediately ---
        chat_text.config(state='normal')
        chat_text.insert(tk.END, f"You: {question}\n")
        chat_text.config(state='disabled')
        chat_text.yview(tk.END)
        question_entry.delete(0, tk.END)
        root.update_idletasks() # Show question before waiting for response

        # --- Disable send button while processing ---
        send_button.config(state=tk.DISABLED)
        question_entry.config(state=tk.DISABLED)

        # --- Get response (Consider threading for long LLM calls) ---
        # Simple blocking call for now:
        try:
             response = ask_cv_question(selected_cv, question)
             chat_text.config(state='normal')
             chat_text.insert(tk.END, f"Chatbot: {response}\n\n") # Add extra newline
             chat_text.config(state='disabled')
             chat_text.yview(tk.END)
        except Exception as e:
             # Handle potential errors during the ask_cv_question call itself
             logging.error(f"Error getting chatbot response: {e}", exc_info=True)
             chat_text.config(state='normal')
             chat_text.insert(tk.END, f"Chatbot Error: Could not get response. Check logs.\nError: {e}\n\n")
             chat_text.config(state='disabled')
             chat_text.yview(tk.END)
        finally:
            # --- Re-enable send button ---
            send_button.config(state=tk.NORMAL)
            question_entry.config(state=tk.NORMAL)


    send_button = ttk.Button(chatbot_frame, text="Send", command=send_question)
    send_button.grid(row=2, column=2, padx=5, pady=5)


    # --- Configure main_frame resizing ---
    main_frame.columnconfigure(1, weight=1) # Allow entry/combobox column to expand
    # Adjust row weights for expansion - Log/Result area and Chatbot area
    main_frame.rowconfigure(4, weight=1)    # Allow results area to expand
    main_frame.rowconfigure(5, weight=1)    # Allow chatbot area to expand


    # Start the Tkinter event loop
    logging.info("GUI Created. Ready for input.")
    root.mainloop()

# Cell 12: Run the application
if __name__ == "__main__":
    # --- Basic Pre-checks ---
    # Check if placeholder values are still present
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: GOOGLE_API_KEY is not set. Please replace 'YOUR_API_KEY_HERE' in the script or set the environment variable.")
        logging.critical("GOOGLE_API_KEY is not set correctly.")
        # Optionally exit or show a GUI error if Tkinter is already imported
        # For a GUI app, it's better to show the error within the GUI if possible (as done in create_gui)
    if MONGODB_URI == "YOUR_MONGODB_URI":
        print("ERROR: MONGODB_URI is not set. Please replace 'YOUR_MONGODB_URI' in the script or set the environment variable.")
        logging.critical("MONGODB_URI is not set correctly.")

    # Attempt to check Tesseract path existence if set explicitly (optional)
    # try:
    #     tesseract_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
    #     if tesseract_cmd and not os.path.exists(tesseract_cmd):
    #         logging.warning(f"Tesseract path set to '{tesseract_cmd}' but file not found. OCR may fail if needed.")
    # except Exception as e:
    #      logging.info(f"Could not check Tesseract path: {e}. Assuming it's in system PATH if needed.")

    # --- Start the GUI ---
    create_gui()