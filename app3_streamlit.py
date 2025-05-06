


import streamlit as st
import subprocess
import sys
import os
import logging

# --- Configuration ---
# Names of the Python script files you want to launch
# IMPORTANT: These files MUST be in the SAME directory as this launcher script,
#            or you need to provide the full path to them.
# UPDATE THESE TO POINT TO THE STREAMLIT VERSIONS IF YOU CONVERT THEM TOO
SECTION1_SCRIPT = "Section1_streamlit.py" # Renamed for clarity
SECTION3_SCRIPT = "Section3_streamlit.py" # Renamed for clarity
    
# Setup basic logging for the launcher itself
# Note: Streamlit manages its own logging, but this can still be useful
#       if running the script directly or for subprocess monitoring.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - LAUNCHER - %(levelname)s - %(message)s')

# --- Functions to Launch Scripts ---

def check_file_exists(script_name):
    """Checks if the specified script file exists in the current directory."""
    if not os.path.isfile(script_name):
        error_msg = f"Error: Script file '{script_name}' not found in the current directory.\nPlease ensure it exists alongside this launcher."
        logging.error(error_msg)
        st.error(error_msg) # Use Streamlit error display
        return False
    return True

def launch_script_streamlit(script_name):
    """
    Attempts to launch the specified Python script using subprocess.
    In a Streamlit context, this typically means starting another Streamlit process.
    """
    if not check_file_exists(script_name):
        return # Stop if file doesn't exist

    try:
        logging.info(f"Attempting to launch script: {script_name} using 'streamlit run'")
        # Command to run another Streamlit app
        # NOTE: This starts a *new* server process. Manage ports if needed.
        command = [sys.executable, "-m", "streamlit", "run", script_name]

        # Use Popen to run in the background (non-blocking)
        process = subprocess.Popen(command)

        logging.info(f"Launched '{script_name}' process with PID: {process.pid}. It might take a moment to start.")
        st.info(f"Attempted to launch '{os.path.basename(script_name)}'. Check your terminal or browser for the new Streamlit app.")
        st.caption(f"(Process PID: {process.pid} using command: {' '.join(command)})")
        # We don't wait for it or manage its lifecycle further here.

    except FileNotFoundError:
        error_msg = f"Error: '{sys.executable}' or 'streamlit' command not found. Cannot launch script. Is Streamlit installed correctly in the environment?"
        logging.error(error_msg)
        st.error(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred while trying to launch '{script_name}':\n{e}"
        logging.error(error_msg, exc_info=True)
        st.error(error_msg)

# --- Streamlit Frontend ---

st.set_page_config(layout="wide", page_title="CV ANALYSIS" )

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        text-align: center;
    }
    
    .card-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
    }
    
    .card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    .card-blue:hover {
        background-color: #e6f2ff;
        border: 2px solid #4da6ff;
    }
    
    .card-green:hover {
        background-color: #e6fff2;
        border: 2px solid #4dffa6;
    }
    
    .card-title {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    .card-icon {
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .card-description {
        font-size: 1.2rem !important;
        text-align: center;
    }
    
    .stButton > button {
        width: 100%;
        height: 4rem;
        font-size: 1.2rem !important;
        font-weight: 500;
        margin-top: 1.5rem;
        border-radius: 8px;
    }
    
    .footer-text {
        font-size: 1.1rem !important;
        text-align: center;
    }
    
    /* Hide the default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">CV Analysis Tool Launcher</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown('<p style="font-size: 1.4rem; text-align: center; margin-bottom: 3rem;">Select a module to launch</p>', unsafe_allow_html=True)

# Create two columns for the cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card-container card-blue">
        <div class="card-icon">🚀</div>
        <h2 class="card-title">Employee Skill Gap Analyzer</h2>
        <p class="card-description">Analyze employee skills and identify gaps compared to job requirements.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch", key="run_section1", use_container_width=True,type = "primary"):
        logging.info("Section 1 button clicked.")
        launch_script_streamlit(SECTION1_SCRIPT)

with col2:
    st.markdown("""
    <div class="card-container card-green">
        <div class="card-icon">📊</div>
        <h2 class="card-title">Recruitment Batch Processing</h2>
        <p class="card-description">Process and analyze multiple candidate CVs for efficient recruitment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch", key="run_section3", use_container_width=True,type = "primary"):
        logging.info("Section 3 button clicked.")
        launch_script_streamlit(SECTION3_SCRIPT)

# Footer
st.markdown("---")
st.markdown('<p class="footer-text">If a new browser tab doesn\'t open automatically, check the terminal where you ran this launcher script for the URL of the newly launched section.</p>', unsafe_allow_html=True)

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("Starting CV Analysis Tool Launcher (Streamlit)...")
    # The Streamlit UI is defined above and runs automatically.
    pass