import streamlit as st
import subprocess
import os, json
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
                        
                        # --- NEW: LOAD SCORES WITH RANK MATCHING ---
                        scores_map = {} # Key: (Variation_ID, Rank_ID), Value: Score Dict
                        
                        if image_list:
                            panel_dir = os.path.dirname(image_list[0])
                            score_file = os.path.join(panel_dir, "scores.json")
                            
                            if os.path.exists(score_file):
                                try:
                                    with open(score_file, "r", encoding="utf-8") as f:
                                        data = json.load(f)
                                        for var in data.get("variations", []):
                                            v_id = var.get("variation_id")
                                            clip = var.get("clip_score", 0.0)
                                            
                                            # Store score for EACH layout rank
                                            for layout in var.get("layout_options", []):
                                                rank = layout.get("rank")
                                                scores_map[(v_id, rank)] = {
                                                    "final": layout.get("final_score", 0),
                                                    "clip": clip,
                                                    "sim": layout.get("sim_score", 0),
                                                    "geom": layout.get("geom_penalty", 0)
                                                }
                                except Exception as e:
                                    st.warning(f"Could not load scores: {e}")

                        # --- DISPLAY IMAGES & SCORES ---
                        cols = st.columns(3)
                        for j, img_path in enumerate(image_list):
                            col = cols[j % 3]
                            
                            # 1. Parse VarID and Rank from filename
                            # Standard format: "{var_id:02d}_name_{rank}_onlyname.png"
                            # Example: "00_name_1_onlyname.png" -> Var 0, Rank 1
                            try:
                                fname = os.path.basename(img_path)
                                parts = fname.split("_")
                                # parts[0] = var_id, parts[1] = "name", parts[2] = rank
                                var_id = int(parts[0])
                                rank_id = int(parts[2]) 
                            except:
                                var_id = -1
                                rank_id = -1

                            # 2. Display Image
                            try:
                                col.image(img_path, caption=fname, use_container_width=True)
                                
                                # 3. Display Specific Score
                                key = (var_id, rank_id)
                                
                                if key in scores_map:
                                    s = scores_map[key]
                                    score_color = "green" if s['final'] > 80 else "orange" if s['final'] > 50 else "red"
                                    
                                    col.markdown(f"""
                                    <div style="text-align:center; background-color:#f0f2f6; padding:5px; border-radius:5px; margin-bottom:10px;">
                                        <strong style="color:{score_color}; font-size:18px;">{s['final']:.1f}</strong><br>
                                        <span style="font-size:12px; color:#555;">
                                            CLIP: <b>{s['clip']:.2f}</b> | Sim: <b>{s['sim']:.2f}</b><br>
                                            Penalty: <span style="color:red">-{s['geom']:.1f}</span>
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # If no specific rank found (maybe manual pipeline run?), check for "Best" (Rank 0 default)
                                    # or just don't show anything to avoid confusion.
                                    pass
                                    
                            except Exception as img_e:
                                col.error(f"Failed: {img_e}")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
        # Clean up the temporary file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)