import streamlit as st
import subprocess
import os
import tempfile
import glob
from datetime import datetime
import re
from collections import defaultdict
import sys  

try:
    from lib.layout.layout import MangaLayout
except ModuleNotFoundError:
    st.error("Error: 'lib' module not found. Please run 'pip install -e .' in your terminal from the project's root directory.")
    st.stop()
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure all requirements are installed: pip install -r requirements.txt")
    st.stop()


st.set_page_config(page_title="MangaGen Pipeline", layout="wide")
st.title("Manga Generation Pipeline  manga-gen 📖")

st.info("This UI runs the main `src/pipeline.py` script. Make sure your Stable Diffusion WebUI server is running with the API enabled.")

# 1. Get script input from the user
st.subheader("1. Enter Your Manga Script")
default_script = 'Scene: A boy and a girl are in a classroom.\nBoy: "Hello! How are you?"\nGirl (surprised): "Oh! Hi. I\'m doing well, thanks."'
script_text = st.text_area("Script:", value=default_script, height=250)

# 2. Add the "Generate" button
if st.button("Generate Panels", type="primary"):
    if not script_text.strip():
        st.warning("Please enter a script before generating.")
    else:
        # Create a temporary file for the script
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(script_text)
            temp_script_path = f.name
        
        # Define output path based on the pipeline's logic
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S") # Added seconds for uniqueness
        output_dir = os.path.join("output", f"ui_run_{date_str}")

        st.subheader("2. Generation Log")
        
        # 3. Call src/pipeline.py as a subprocess
        command = [
            sys.executable, 
            "src/pipeline.py",
            "--script_path", temp_script_path,
            "--output_path", output_dir
        ]

        try:
            # Set up the environment to force UTF-8 (for Windows)
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            
            with st.spinner(f"Running pipeline... This will take a few minutes. Check your server logs for progress."):
                with st.expander("Show Console Log", expanded=True): # Make it open by default
                    st.write(f"Running command: `{' '.join(command)}`")

                    # --- THIS IS THE NEW LOGIC ---
                    
                    # Create a placeholder for the live log text
                    log_placeholder = st.empty()
                    log_output = "" # A string to accumulate the log

                    # Start the subprocess with Popen
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, # Combine stdout and stderr
                        text=True,
                        encoding='utf-8',
                        env=my_env,
                        bufsize=1 # Use line-buffering
                    )

                    # Read the output line by line, in real-time
                    # The 'for' loop will automatically wait for new lines
                    for line in process.stdout:
                        log_output += line
                        log_placeholder.text_area("Live Console Log:", value=log_output, height=300)

                    # Wait for the process to finish
                    process.wait()
                    # --- END OF NEW LOGIC ---

                    # Check the final return code
                    if process.returncode != 0:
                        st.error("The pipeline failed. See the final log above for details.")
                        st.stop() # Stop the app if the pipeline failed

            st.success("Pipeline finished successfully!")

            # 4. Find and display the output images
            st.subheader("3. Generated Panels")
            image_dir = os.path.join(output_dir, "images")
            
            # Find all '...name...' pngs, which are the final output
            search_path = os.path.join(output_dir, "*", "images", "panel*", "*_onlyname.png")
            image_files = glob.glob(search_path)
            
            panel_groups = defaultdict(list)
            panel_pattern = re.compile(r"(panel\d+)") # Regex to find "panelXXX"

            for img_path in image_files:
                match = panel_pattern.search(img_path)
                if match:
                    panel_name = match.group(1)
                    panel_groups[panel_name].append(img_path)

            if not panel_groups:
                st.warning("Pipeline ran, but no output images were found. Check the log above for errors.")
            else:
                # Create a tab for each panel group
                panel_names = sorted(panel_groups.keys())
                tabs = st.tabs(panel_names)
                
                for i, panel_name in enumerate(panel_names):
                    with tabs[i]:
                        st.subheader(f"Variations for {panel_name}")
                        image_list = panel_groups[panel_name]
                        cols = st.columns(3)
                        for j, img_path in enumerate(image_list):
                            col = cols[j % 3]
                            try:
                                col.image(img_path, caption=os.path.basename(img_path), width='stretch')
                            except Exception as img_e:
                                col.error(f"Failed to load {os.path.basename(img_path)}: {img_e}")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
        # Clean up the temporary file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)