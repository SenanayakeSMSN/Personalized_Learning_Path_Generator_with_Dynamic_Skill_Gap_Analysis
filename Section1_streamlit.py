
# Cell 2: Import Libraries and Setup
import streamlit as st
import os
import json
import logging
import re
import time
import tempfile 

from pymongo import MongoClient
# Removed tkinter imports
import google.generativeai as genai
import pdfplumber
import pytesseract
from PIL import Image
from google.api_core.exceptions import ResourceExhausted

# Setup logging
# Streamlit handles basic logging, but configuring can add more detail if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - S1 - %(levelname)s - %(message)s')

API_KEY_POOL = [
    st.secrets["secrets"]["API_KEY_1"],
    st.secrets["secrets"]["API_KEY_2"],
    st.secrets["secrets"]["API_KEY_3"]
]
MONGODB_URI = st.secrets["secrets"]["MONGODB_URI"]
# --- State variables for API key rotation ---
# Use Streamlit session state to maintain index across reruns
if 'current_api_key_index' not in st.session_state:
    st.session_state.current_api_key_index = 0
# Helper function to configure Gemini with retry and key switching
def configure_gemini_with_retry():
    max_retries = 3
    retry_delays = [5, 10, 20]  # Seconds

    for attempt in range(max_retries + 1):
        current_index = st.session_state.current_api_key_index
        try:
            if current_index >= len(API_KEY_POOL):
                st.error("All API keys exhausted. Please add more keys or wait for rate limit reset.")
                logging.error("All API keys exhausted.")
                return False # Indicate failure clearly

            api_key_to_use = API_KEY_POOL[current_index]
            genai.configure(api_key=api_key_to_use)
            logging.info(f"Configured Gemini with API key index {current_index}")
            return True # Indicate success

        except ResourceExhausted as e:
            logging.warning(f"Rate limit exceeded with API key index {current_index}. Error: {e}")
            if attempt < max_retries:
                wait_time = retry_delays[attempt]
                logging.warning(f"Retrying in {wait_time} seconds...")
                # In Streamlit, time.sleep blocks execution, use with caution or show spinner
                time.sleep(wait_time) # Simple approach for now
            else:
                logging.warning(f"Retries exhausted for key index {current_index}. Switching to next key.")
                st.session_state.current_api_key_index += 1
                if st.session_state.current_api_key_index >= len(API_KEY_POOL):
                    st.error("All API keys have been exhausted after retries.")
                    logging.error("All API keys exhausted after retries.")
                    return False # Failed after switching too
                # If switched, retry immediately in the next loop iteration
        except Exception as e:
            logging.error(f"Error configuring Gemini with key index {current_index}: {str(e)}")
            st.session_state.current_api_key_index += 1 # Try next key on general error too
            if st.session_state.current_api_key_index >= len(API_KEY_POOL):
                 st.error("All API keys exhausted due to configuration errors.")
                 logging.error("All API keys exhausted due to configuration errors.")
                 return False
            # Continue loop to try next key

    # If loop finishes without returning True
    st.error("Failed to configure Gemini after all attempts and key switches.")
    logging.error("Failed to configure Gemini after all attempts.")
    return False


# Cell 3: MongoDB Connection (Cached for Streamlit efficiency)
@st.cache_resource(ttl=3600) # Cache resource for 1 hour
def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        db = client["skillgapanalysis"]
        collection = db["jobrole_skill"]
       
        logging.info("Connected to MongoDB Atlas")
        return collection
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        # Raise error to prevent app from running without DB
        st.error(f"Failed to connect to MongoDB: {str(e)}. Please check URI and network.")
        st.stop() # Stop execution if DB connection fails
        # return None # Return None if using st.cache_data

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_job_roles(_collection): # Pass collection explicitly for caching
    if _collection is None:
        return []
    try:
        # Ensure collection is valid (simple check)
        if "jobrole_skill" not in _collection.database.list_collection_names():
             logging.error("jobrole_skill collection not found in DB.")
             return []
        job_roles_cursor = _collection.find({}, {"Job_Role": 1})
        job_roles = [doc["Job_Role"] for doc in job_roles_cursor if "Job_Role" in doc and doc["Job_Role"]]
        job_roles = sorted(list(set(job_roles))) # Remove duplicates and sort
        logging.info(f"Retrieved {len(job_roles)} job roles.")
        return job_roles
    except Exception as e:
        logging.error(f"Error retrieving job roles: {str(e)}")
        st.warning(f"Could not retrieve job roles: {e}")
        return []

# Cell 4: Enhanced CV Text Extraction

def extract_cv_text(cv_path):
    try:
        with pdfplumber.open(cv_path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n\n"
                    logging.info(f"Extracted text from page {page_number} using pdfplumber")
                else:
                    logging.warning(f"No text extracted from page {page_number}. Trying OCR.")
                    try:
                        # Increase resolution for potentially better OCR
                        img = page.to_image(resolution=300).original
                        # Ensure pytesseract path is set if needed (usually done system-wide or via env var)
                        ocr_text = pytesseract.image_to_string(img)
                        text += ocr_text + "\n\n"
                        logging.info(f"Extracted OCR text from page {page_number}")
                    except ImportError:
                         logging.error("OCR skipped: pytesseract or Pillow not installed.")
                         text += "\n\n" # Keep spacing consistent
                    except Exception as ocr_e:
                        logging.error(f"OCR failed for page {page_number}: {str(ocr_e)}")
                        text += "\n\n" # Keep spacing consistent
            if not text.strip():
                # Return empty string instead of raising error, let caller handle
                logging.warning(f"No text extracted from CV: {os.path.basename(cv_path)}")
                return ""
            logging.info(f"Extracted text from CV: {os.path.basename(cv_path)} ({len(text)} chars)")
            return text
    except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
        logging.error(f"PDF is password-protected: {os.path.basename(cv_path)}")
        st.error(f"Failed to read CV '{os.path.basename(cv_path)}': PDF is password-protected.")
        return None # Indicate error
    except FileNotFoundError:
         logging.error(f"CV file not found at path: {cv_path}")
         st.error(f"Internal error: CV file path not found: {cv_path}")
         return None # Indicate error
    except Exception as e:
        logging.error(f"Error reading CV '{os.path.basename(cv_path)}': {str(e)}")
        st.error(f"Error reading CV '{os.path.basename(cv_path)}': {str(e)}")
        return None # Indicate error

# Cell 5: Gemini Skill and Experience Extraction (Function logic unchanged)
def extract_skills_with_gemini(cv_text):
    # Default return structure for errors
    error_result = [], {}, "0 years 0 months"
    if not cv_text or not cv_text.strip():
        logging.error("CV text is empty. Cannot extract skills.")
        return error_result

    try:
        if not configure_gemini_with_retry():
            st.error("Failed to configure Gemini: All API keys exhausted.")
            return error_result # Return default error structure

        model = genai.GenerativeModel("gemini-1.5-flash") # Ensure model name is correct
        prompt = (
             # Keep the detailed prompt from the original file
            "You are an expert in CV analysis. Perform the following steps on the provided CV text:\n"
            "1. Parse the CV text into a structured JSON object with the following fields (include only fields that can be identified, leave empty if not found):\n"
            "   - name: The candidate's full name (string).\n"
            "   - education: List of education entries (array of strings, e.g., ['B.S. Computer Science, XYZ University, 2020']).\n"
            "   - projects: List of project descriptions (array of strings, e.g., ['Built a web app using Django']).\n"
            "   - experience: List of work experience entries with dates from the experience section only (array of strings, e.g., ['Software Engineer at ABC Corp, Jan 2019 - Dec 2021']). Identify the experience section by headings like 'Professional Experience', 'Work Experience', 'Working Experience', 'Employment History', or similar. Exclude internships or roles mentioned in projects or education.\n"
            "   - skills: List of explicit skills from a 'Skills' or similar section (array of strings, e.g., ['Python', 'Java']).\n"
            "   - contact: Contact information (string, e.g., 'email: john@example.com').\n"
            "   - experience_duration: Total work experience duration as a string, calculated as follows:\n"
            "     - Use only the 'experience' section identified by headings like 'Professional Experience', 'Work Experience', etc.\n"
            "     - If the experience section is missing or empty, return '0 years 0 months'.\n"
            "     - Parse dates in various formats, including 'MM/YYYY' (e.g., '02/2024'), 'Month YYYY' (e.g., 'June 2018'), 'YYYY - YYYY' (e.g., '2019 - 2021'), or 'MM-YYYY' (e.g., '02-2024'). For 'current' or 'Present', assume today's date: April 19, 2025.\n" # Hardcoded date from original, consider making dynamic
            "     - For partial dates (e.g., '2019 - 2021'), assume January for the start year and December for the end year (e.g., 'Jan 2019 - Dec 2021').\n"
            "     - Identify the earliest start date (first appointed job) and the latest end date (last appointed job) across all experience entries.\n"
            "     - Subtract the earliest start date from the latest end date to calculate the total duration in years and months (e.g., '2 years 3 months', '0 years 1 months', '3 years 0 months').\n"
            "     - Use plural forms consistently (e.g., '1 years', '0 months') for uniformity.\n"
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
            
            "Example 1 (multiple jobs, varied date formats):\n"
            "Input: 'Name: John Doe\\nProfessional Experience: Software Engineer at ABC Corp, 02/2019 - 12/2021; Data Scientist at XYZ Inc, June 2022 - current\\nSkills: Python, ML'\n"
            "Output: {\\\"structured_cv\\\":{\\\"name\\\":\\\"John Doe\\\",\\\"skills\\\":[\\\"Python\\\",\\\"ML\\\"],\\\"experience\\\":[\\\"Software Engineer at ABC Corp, 02/2019 - 12/2021\\\",\\\"Data Scientist at XYZ Inc, June 2022 - current\\\"],\\\"projects\\\":[],\\\"education\\\":[],\\\"contact\\\":\\\"\\\",\\\"experience_duration\\\":\\\"6 years 2 months\\\"},\\\"skills\\\":[\\\"Python\\\",\\\"Machine Learning\\\"],\\\"experience_duration\\\":\\\"6 years 2 months\\\"}\n"
            "Example 2 (short duration):\n"
            "Input: 'Name: Alice Smith\\nWork History: Intern at Tech Inc, 02/2024 - 03/2024'\n"
            "Output: {\\\"structured_cv\\\":{\\\"name\\\":\\\"Alice Smith\\\",\\\"skills\\\":[],\\\"experience\\\":[\\\"Intern at Tech Inc, 02/2024 - 03/2024\\\"],\\\"projects\\\":[],\\\"education\\\":[],\\\"contact\\\":\\\"\\\",\\\"experience_duration\\\":\\\"0 years 1 months\\\"},\\\"skills\\\":[],\\\"experience_duration\\\":\\\"0 years 1 months\\\"}\n"
         
            "\n\nCV Text:\n" + cv_text
        )

        max_retries = 3
        retry_delays = [5, 10, 20]
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting Gemini call (Attempt {attempt + 1}/{max_retries}) for skill extraction...")
                response = model.generate_content(prompt)

                # Check for blocked response
                if not response.parts:
                     block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                     block_message = response.prompt_feedback.block_reason_message if response.prompt_feedback else "No message"
                     logging.error(f"Gemini response blocked. Reason: {block_reason}, Message: {block_message}")
                     # Optionally, retry or switch keys if block is due to safety/rate limit
                     # For simplicity now, return error
                     st.error(f"Content generation blocked by API. Reason: {block_reason}. Try modifying the CV or prompt if applicable.")
                     return error_result

                raw_response = response.text.strip()
                logging.info(f"Gemini raw response (Skill Extraction): {raw_response[:150]}...") # Log more context

                # Clean potential markdown fences more robustly
                cleaned_response = re.sub(r"^```json\s*", "", raw_response, flags=re.IGNORECASE)
                cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
                cleaned_response = cleaned_response.strip()
                logging.info(f"Cleaned Gemini response: {cleaned_response[:150]}...")

                if not cleaned_response:
                     logging.error("Gemini returned an empty response after cleaning.")
                     # Could retry here, or return error
                     continue # Retry the Gemini call

                # Attempt to parse JSON
                result = json.loads(cleaned_response)

                # Validate structure
                if not isinstance(result, dict) or "structured_cv" not in result or "skills" not in result or "experience_duration" not in result:
                    logging.error("Gemini response is not a valid JSON object with required fields.")
                    # Consider retry or return error
                    continue # Retry the Gemini call

                # Extract and validate types
                structured_cv = result.get("structured_cv", {})
                skills = result.get("skills", [])
                experience_duration = result.get("experience_duration", "0 years 0 months")

                if not isinstance(structured_cv, dict): structured_cv = {}
                if not isinstance(skills, list): skills = []
                if not isinstance(experience_duration, str): experience_duration = "0 years 0 months"

                logging.info(f"Structured CV Keys: {list(structured_cv.keys())}")
                logging.info(f"Extracted skills count: {len(skills)}")
                logging.info(f"Extracted experience duration: {experience_duration}")
                return skills, structured_cv, experience_duration # Success

            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse cleaned Gemini response as JSON: {cleaned_response[:200]}, Error: {str(e)}")
                # Don't immediately return error, retry the Gemini call
                if attempt < max_retries - 1:
                     logging.warning(f"Retrying Gemini call due to JSON decode error...")
                     time.sleep(retry_delays[attempt]) # Wait before retry
                     continue
                else:
                     logging.error("Max retries reached for JSON decode error.")
                     st.error(f"Failed to process API response (JSON Error): {e}. Please try again.")
                     return error_result

            except ResourceExhausted as e:
                 logging.warning(f"Rate limit hit during skill extraction (Attempt {attempt+1}).")
                 if attempt < max_retries - 1:
                     wait_time = retry_delays[attempt]
                     logging.warning(f"Retrying in {wait_time} seconds...")
                     time.sleep(wait_time)
                 else:
                     logging.warning(f"Retries exhausted for rate limit. Switching API key.")
                     st.session_state.current_api_key_index += 1
                     if not configure_gemini_with_retry(): # Configure with the *new* index
                         st.error("Failed to analyze CV: All API keys exhausted after rate limit.")
                         return error_result # Failed after switching
                     # If configure successful, the loop continues to the next attempt with the new key/model
                     model = genai.GenerativeModel("gemini-1.5-flash") # Re-instantiate model with new config if needed
            except Exception as e:
                 logging.error(f"Unexpected error in Gemini skill extraction (Attempt {attempt+1}): {str(e)}", exc_info=True)
                 # Consider retry for transient errors, but for now return error on general exception
                 st.error(f"An unexpected error occurred during skill extraction: {e}")
                 return error_result

        # If loop completes without success
        logging.error("Failed to extract skills from Gemini after all retries and key switches.")
        st.error("Failed to get valid skill data from the API after multiple attempts.")
        return error_result

    except Exception as e:
        # Catch errors before the loop (e.g., initial configure_gemini call)
        logging.error(f"Error setting up Gemini skill extraction: {str(e)}")
        st.error(f"Error setting up Gemini analysis: {e}")
        return error_result


# Cell 6: Gemini Custom Question Answering (Function logic unchanged)
def answer_cv_question(cv_text, structured_cv, question, skill_gap_result=None, experience_duration="0 years 0 months"):
    if not cv_text or not structured_cv:
         logging.warning("Cannot answer question without CV text and structured data.")
         return "Error: CV data is not available for context."
    if not question:
         logging.warning("Cannot answer empty question.")
         return "Error: No question provided."

    try:
        if not configure_gemini_with_retry():
            st.error("Failed to configure Gemini for question answering.")
            return "Error: Could not connect to the analysis service."

        model = genai.GenerativeModel("gemini-1.5-flash")
        context = f"CV Text Snippet (first 1000 chars):\n{cv_text[:1000]}\n\nStructured CV Data:\n{json.dumps(structured_cv, indent=2)}\n\nTotal Work Experience: {experience_duration}\n\n"
        if skill_gap_result:
            context += f"Skill Gap Analysis Result:\nJob Role: {skill_gap_result.get('job_role', 'N/A')}\nCV Skills: {', '.join(skill_gap_result.get('cv_skills', []))}\nRequired Skills: {', '.join(skill_gap_result.get('required_skills', []))}\nMissing Skills: {', '.join(skill_gap_result.get('missing_skills', []))}\n\n"

        prompt = (
            "You are an expert career advisor. Using the provided CV text snippet, structured CV data, total work experience, and skill gap analysis (if available), answer the following question concisely, professionally, and in a personalized manner. "
            "If asked for a learning path, provide actionable steps, suggest types of resources (like specific online courses, projects), and a realistic timeframe based on the context. "
            "Return the answer as plain text, suitable for direct display.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
            "Answer:"
        )

        max_retries = 3
        retry_delays = [5, 10, 20]
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting Gemini call (Attempt {attempt + 1}/{max_retries}) for question answering...")
                response = model.generate_content(prompt)

                # Check for blocked response
                if not response.parts:
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    block_message = response.prompt_feedback.block_reason_message if response.prompt_feedback else "No message"
                    logging.error(f"Gemini response blocked (Question Answering). Reason: {block_reason}, Message: {block_message}")
                    st.error(f"Content generation blocked by API. Reason: {block_reason}.")
                    return f"Error: Could not generate answer due to API restrictions ({block_reason})."

                answer = response.text.strip()
                logging.info(f"Gemini answered question '{question[:50]}...': {answer[:100]}...")
                return answer # Success

            except ResourceExhausted as e:
                logging.warning(f"Rate limit hit during question answering (Attempt {attempt+1}).")
                if attempt < max_retries - 1:
                    wait_time = retry_delays[attempt]
                    logging.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.warning(f"Retries exhausted for rate limit. Switching API key.")
                    st.session_state.current_api_key_index += 1
                    if not configure_gemini_with_retry():
                         st.error("Failed to answer question: All API keys exhausted after rate limit.")
                         return "Error: Analysis service unavailable (rate limit)."
                    # Re-instantiate model if needed after config change
                    model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception as e:
                logging.error(f"Error answering question '{question[:50]}...' with Gemini: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred while answering the question: {e}")
                return f"Error answering question: {str(e)}"

        # If loop completes without success
        logging.error("Failed to answer question after all retries and key switches.")
        st.error("Failed to get an answer from the API after multiple attempts.")
        return "Error: Could not get an answer from the analysis service."

    except Exception as e:
         logging.error(f"Error setting up Gemini question answering: {str(e)}")
         st.error(f"Error setting up question answering service: {e}")
         return "Error: Could not initialize the question answering service."


# Cell 7: MongoDB Query (Function logic unchanged)
def get_required_skills(job_collection, job_role):
    if job_collection is None:
        st.error("Database connection not available for fetching required skills.")
        return None # Indicate error
    try:
        # Use case-insensitive query for flexibility
        job_doc = job_collection.find_one({"Job_Role": {"$regex": f"^{re.escape(job_role)}$", "$options": "i"}})
        if not job_doc:
            logging.warning(f"Job role '{job_role}' not found in database.")
            st.warning(f"Job role '{job_role}' not found in the database. Cannot determine required skills.")
            return [] # Return empty list, not None, to indicate role found but no skills defined or role missing
        required_skills_raw = job_doc.get("Required_Skills")
        if not required_skills_raw or not isinstance(required_skills_raw, str):
             logging.warning(f"Required_Skills field missing or not a string for job role '{job_role}'.")
             return [] # Treat as no skills defined for this role

        required_skills = [skill.strip() for skill in required_skills_raw.split(",") if skill.strip()]
        logging.info(f"Required skills for {job_role}: {required_skills}")
        return required_skills
    except Exception as e:
        logging.error(f"Error retrieving required skills for '{job_role}': {str(e)}")
        st.error(f"Database error retrieving required skills for '{job_role}': {e}")
        return None # Indicate error


# Cell 8: Skill Gap Analysis (Function logic unchanged)
def skill_gap_analysis(cv_skills, required_skills):
    try:
        # Normalize skills to lower case for case-insensitive comparison
        cv_skills_set = set(skill.lower().strip() for skill in cv_skills if skill)
        required_skills_set = set(skill.lower().strip() for skill in required_skills if skill)

        missing_skills_lower = required_skills_set - cv_skills_set

      
        required_lower_to_original = {skill.lower().strip(): skill for skill in required_skills if skill}
        missing_skills_original_case = [required_lower_to_original[lower_skill]
                                       for lower_skill in missing_skills_lower
                                       if lower_skill in required_lower_to_original]

        logging.info(f"Missing skills (Original Case): {missing_skills_original_case}")
        return sorted(missing_skills_original_case) # Sort for consistent output
    except Exception as e:
        logging.error(f"Error in skill gap analysis: {str(e)}")
        st.error(f"Error performing skill gap analysis: {e}")
        # Return None or empty list depending on how you want to handle errors downstream
        return None # Indicate error


# Cell 9: Streamlit GUI
st.set_page_config(layout="wide")
st.title("📄 Employee Skill Gap Analyzer")

# --- Initialize Session State ---
# Store CV info, analysis results, etc.
if 'cv_uploaded' not in st.session_state:
    st.session_state.cv_uploaded = False
    st.session_state.cv_filename = None
    st.session_state.cv_temp_path = None
    st.session_state.cv_text = None
    st.session_state.structured_cv = None
    st.session_state.skill_gap_result = None
    st.session_state.analysis_done = False
    st.session_state.analysis_error = None
    st.session_state.question_answer = None
    st.session_state.learning_path = None

# --- MongoDB Connection and Job Roles ---
job_collection = connect_to_mongodb()
job_roles = get_job_roles(job_collection)
if not job_roles:
    st.warning("No job roles loaded from the database. Analysis requires a target role.")
   

# --- GUI Layout ---
col1, col2 = st.columns([2, 3]) # Left column for controls, Right for results

with col1:
    st.header("Controls")

    # --- CV Upload ---
    uploaded_file = st.file_uploader("Upload CV (PDF only)", type=["pdf"], key="cv_uploader")

    if uploaded_file is not None:
        # Check if it's a new file upload
        if not st.session_state.cv_uploaded or st.session_state.cv_filename != uploaded_file.name:
            st.info(f"Processing uploaded file: {uploaded_file.name}")
            # Save to a temporary file to get a path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.cv_temp_path = tmp_file.name
            st.session_state.cv_filename = uploaded_file.name
            st.session_state.cv_uploaded = True
            st.session_state.analysis_done = False # Reset analysis on new upload
            st.session_state.skill_gap_result = None
            st.session_state.structured_cv = None
            st.session_state.cv_text = None
            st.session_state.analysis_error = None
            st.session_state.question_answer = None
            st.session_state.learning_path = None
            st.success(f"CV '{uploaded_file.name}' uploaded successfully.")
            logging.info(f"Uploaded CV: {uploaded_file.name}, Temp Path: {st.session_state.cv_temp_path}")

            # Automatically extract text after upload
            with st.spinner("Extracting text from CV..."):
                 cv_text_extracted = extract_cv_text(st.session_state.cv_temp_path)
                 if cv_text_extracted is None: # Indicates an error during extraction
                      st.error(f"Failed to extract text from {st.session_state.cv_filename}. Cannot proceed.")
                      # Clean up state for failed extraction
                      st.session_state.cv_uploaded = False
                      st.session_state.cv_text = None
                      # Clean up temp file if extraction failed
                      if st.session_state.cv_temp_path and os.path.exists(st.session_state.cv_temp_path):
                          os.remove(st.session_state.cv_temp_path)
                          st.session_state.cv_temp_path = None
                 elif not cv_text_extracted: # Empty text extracted
                      st.warning(f"No text could be extracted from {st.session_state.cv_filename}. Analysis might be inaccurate.")
                      st.session_state.cv_text = "" # Store empty string
                 else:
                      st.session_state.cv_text = cv_text_extracted
                      logging.info(f"Successfully extracted text for {st.session_state.cv_filename}.")
                      st.info("CV text extracted. Select a job role and click Analyze.")

    # Display current CV status
    if st.session_state.cv_uploaded and st.session_state.cv_filename:
        st.info(f"Current CV: **{st.session_state.cv_filename}**")
        if st.button("Clear Current CV", key="clear_cv"):
            if st.session_state.cv_temp_path and os.path.exists(st.session_state.cv_temp_path):
                os.remove(st.session_state.cv_temp_path)
                logging.info(f"Removed temp file: {st.session_state.cv_temp_path}")
            st.session_state.cv_uploaded = False
            st.session_state.cv_filename = None
            st.session_state.cv_temp_path = None
            st.session_state.cv_text = None
            st.session_state.structured_cv = None
            st.session_state.skill_gap_result = None
            st.session_state.analysis_done = False
            st.session_state.analysis_error = None
            st.session_state.question_answer = None
            st.session_state.learning_path = None
            # Clear the file uploader widget state requires rerunning the script
            st.rerun()
    else:
        st.info("No CV uploaded.")

    # --- Job Role Selection ---
    # Add a "Select Role" option
    job_role_options = ["Select a Job Role"] + job_roles if job_roles else ["No roles found"]
    selected_job_role = st.selectbox(
        "Target Job Role:",
        options=job_role_options,
        index=0, # Default to "Select a Job Role"
        disabled=(not st.session_state.cv_uploaded or not job_roles) # Disable if no CV or no roles
    )

    # --- Analyze Button ---
    analyze_disabled = (
        not st.session_state.cv_uploaded or
        selected_job_role == "Select a Job Role" or
        selected_job_role == "No roles found" or
        st.session_state.cv_text is None # Disable if text extraction failed
    )
    if st.button("Analyze Skill Gap", key="analyze", disabled=analyze_disabled):
        st.session_state.analysis_done = False
        st.session_state.skill_gap_result = None
        st.session_state.structured_cv = None
        st.session_state.analysis_error = None
        st.session_state.question_answer = None # Clear previous answers
        st.session_state.learning_path = None # Clear previous path

        if st.session_state.cv_text == "":
            st.warning("CV text is empty, analysis may be inaccurate.")
            # Allow analysis to proceed, Gemini might still find something or fail gracefully

        with st.spinner(f"Analyzing CV for '{selected_job_role}' role... (This may take a moment)"):
            try:
                # 1. Extract Skills & Experience with Gemini
                logging.info("Calling Gemini for skill extraction...")
                cv_skills, structured_cv, experience_duration = extract_skills_with_gemini(st.session_state.cv_text)
                st.session_state.structured_cv = structured_cv # Store structured data

                if structured_cv is None or cv_skills is None or experience_duration is None:
                     # Error is likely handled and logged inside extract_skills_with_gemini
                     # If it returns None, it means a critical failure
                     st.error("Failed to extract necessary data from the CV using the analysis service.")
                     st.session_state.analysis_error = "Data extraction failed."
                elif not cv_skills and not structured_cv.get("skills"): # If Gemini explicitly returns no skills
                     logging.warning(f"No skills found in CV '{st.session_state.cv_filename}' by Gemini.")
                     st.warning("No skills were identified in the CV. Skill gap analysis might be incomplete.")
                     # Proceed, but expect empty results for skills

                # 2. Get Required Skills from DB
                logging.info(f"Fetching required skills for {selected_job_role}...")
                required_skills = get_required_skills(job_collection, selected_job_role)

                if required_skills is None: # Indicates DB error
                    st.error("Failed to retrieve required skills from the database.")
                    st.session_state.analysis_error = "Database error fetching required skills."
                else:
                    # 3. Perform Skill Gap Analysis
                    logging.info("Performing skill gap analysis...")
                    missing_skills = skill_gap_analysis(cv_skills, required_skills)

                    if missing_skills is None: # Indicates analysis error
                        st.error("An error occurred during the skill gap calculation.")
                        st.session_state.analysis_error = "Skill gap calculation failed."
                    else:
                        # 4. Store Results in Session State
                        st.session_state.skill_gap_result = {
                            "job_role": selected_job_role,
                            "cv_skills": cv_skills, # Skills extracted by Gemini
                            "required_skills": required_skills,
                            "missing_skills": missing_skills,
                            "experience_duration": experience_duration
                        }
                        st.session_state.analysis_done = True
                        logging.info(f"Skill gap analysis completed for {selected_job_role}")
                        st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {str(e)}")
                st.session_state.analysis_error = f"Unexpected analysis error: {str(e)}"
                logging.error(f"Analysis error for {selected_job_role}: {str(e)}", exc_info=True)


    st.divider()

    # --- Custom Question ---
    st.subheader("Ask a Question about the CV")
    question_disabled = not st.session_state.analysis_done or st.session_state.analysis_error is not None
    custom_question = st.text_input(
        "Enter your question:",
        key="question_input",
        placeholder="e.g., Suggest relevant certifications.",
        disabled=question_disabled
    )

    if st.button("Submit Question", key="ask_question", disabled=(question_disabled or not custom_question)):
        st.session_state.question_answer = None # Clear previous answer
        st.session_state.learning_path = None # Clear learning path if asking general q
        with st.spinner("Getting answer..."):
            try:
                answer = answer_cv_question(
                    st.session_state.cv_text,
                    st.session_state.structured_cv,
                    custom_question,
                    st.session_state.skill_gap_result, # Pass full result for context
                    st.session_state.skill_gap_result.get("experience_duration", "0 years 0 months")
                )
                st.session_state.question_answer = answer
                logging.info("Received answer for custom question.")
            except Exception as e:
                st.error(f"Failed to get answer: {e}")
                logging.error(f"Error answering question: {e}", exc_info=True)


    # --- Generate Learning Path Button ---
    if st.button("Generate Learning Path for Missing Skills", key="gen_path", disabled=question_disabled):
        st.session_state.question_answer = None # Clear general answer
        st.session_state.learning_path = None # Clear previous path
        if not st.session_state.skill_gap_result or not st.session_state.skill_gap_result.get('missing_skills'):
            st.warning("No missing skills identified or analysis not run. Cannot generate learning path.")
        else:
            missing_skills_str = ', '.join(st.session_state.skill_gap_result['missing_skills'])
            job_role_context = st.session_state.skill_gap_result['job_role']
            experience_context = st.session_state.skill_gap_result.get('experience_duration', 'unknown')
            path_question = (
                f"Generate a personalized learning path to acquire the missing skills ({missing_skills_str}) "
                f"for the job role '{job_role_context}', considering the candidate has approximately {experience_context} of relevant work experience. "
                "Provide specific, actionable steps, suggest types of resources (e.g., online courses, project ideas), and estimate a realistic timeline."
            )

            with st.spinner("Generating learning path..."):
                 try:
                     learning_path_answer = answer_cv_question(
                         st.session_state.cv_text,
                         st.session_state.structured_cv,
                         path_question, # Use the constructed question
                         st.session_state.skill_gap_result,
                         experience_context
                     )
                     st.session_state.learning_path = learning_path_answer
                     logging.info("Received learning path.")
                 except Exception as e:
                     st.error(f"Failed to generate learning path: {e}")
                     logging.error(f"Error generating learning path: {e}", exc_info=True)

# --- Results Display Area ---
with col2:
    st.header("Results")

    if not st.session_state.cv_uploaded:
        st.info("Upload a CV and select a job role to begin analysis.")
    elif st.session_state.analysis_error:
        st.error(f"Analysis failed: {st.session_state.analysis_error}")
    elif not st.session_state.analysis_done:
        if st.session_state.cv_text is not None:
             st.info("Click 'Analyze Skill Gap' to process the uploaded CV.")
        # If cv_text is None, an error message was shown during upload/extraction
    else:
        # Display Skill Gap Analysis Results
        if st.session_state.skill_gap_result:
            res = st.session_state.skill_gap_result
            st.subheader(f"Skill Gap Analysis for: {res['job_role']}")
            st.markdown(f"**Total Relevant Work Experience:** {res.get('experience_duration', 'N/A')}")

            expander_skills = st.expander("Skills Details")
            with expander_skills:
                st.markdown("**CV Skills Found:**")
                if res['cv_skills']:
                    st.write(f"`{', '.join(res['cv_skills'])}`")
                else:
                    st.write("None identified.")

                st.markdown("**Required Skills for Role:**")
                if res['required_skills']:
                    st.write(f"`{', '.join(res['required_skills'])}`")
                else:
                    st.write("None specified for this role.")

                st.markdown("**Missing Skills:**")
                if res['missing_skills']:
                    st.warning(f"`{', '.join(res['missing_skills'])}`")
                else:
                    st.success("No missing skills identified!")

            # Display Structured CV Data (Optional)
            if st.session_state.structured_cv:
                 expander_cv_data = st.expander("Extracted CV Data (from Gemini)")
                 with expander_cv_data:
                      st.json(st.session_state.structured_cv)


        # Display Custom Question Answer
        if st.session_state.question_answer:
            st.divider()
            st.subheader("Answer to Your Question")
            st.markdown(st.session_state.question_answer) # Use markdown for better formatting potentially

        # Display Learning Path
        if st.session_state.learning_path:
            st.divider()
            st.subheader("Generated Learning Path")
            st.markdown(st.session_state.learning_path) # Use markdown

