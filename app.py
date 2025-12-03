import streamlit as st
import subprocess
import os
import json
import tempfile
import glob
from datetime import datetime
import re
from collections import defaultdict
import sys

# --- 1. SETUP & IMPORTS ---
try:
    from lib.layout.layout import MangaLayout
except ModuleNotFoundError:
    st.error("Error: 'lib' module not found. Please run 'pip install -e .' in your terminal.")
    st.stop()
except ImportError as e:
    st.error(f"Error importing: {e}. Did you run 'pip install -r requirements.txt'?")
    st.stop()

st.set_page_config(page_title="MangaGen Pipeline", layout="wide")
st.title("Manga Generation Pipeline 📖")

# --- 2. SESSION STATE MANAGEMENT ---
if 'current_view_path' not in st.session_state:
    st.session_state['current_view_path'] = None
if 'generation_log' not in st.session_state:
    st.session_state['generation_log'] = ""

# Ensure output folder exists
OUTPUT_ROOT = "output"
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

# --- 3. GENERATION INPUT ---
st.subheader("1. New Generation")
default_script = 'Scene: A boy and a girl are in a classroom.\nBoy: "Hello! How are you?"\nGirl (surprised): "Oh! Hi. I\'m doing well, thanks."'
script_text = st.text_area("Script:", value=default_script, height=150)

if st.button("🚀 Generate Panels", type="primary"):
    if not script_text.strip():
        st.warning("Please enter a script first.")
    else:
        # A. Prepare File
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(script_text)
            temp_script_path = f.name
        
        # B. Prepare Output Path
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_ROOT, f"ui_run_{date_str}")

        st.info("Running pipeline... (Logs will appear below)")
        
        # C. Run Pipeline
        command = [
            sys.executable, 
            "src/pipeline.py",
            "--script_path", temp_script_path,
            "--output_path", output_dir
        ]
        
        try:
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            
            # Temporary placeholder for live feedback
            live_log_placeholder = st.empty()
            full_log_buffer = ""

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=my_env,
                bufsize=1
            )

            # Capture logs in real-time
            for line in process.stdout:
                full_log_buffer += line
                # Update the live display
                live_log_placeholder.text_area("Executing...", value=full_log_buffer, height=300)

            process.wait()

            # D. SAVE LOGS & STATE
            st.session_state['generation_log'] = full_log_buffer

            if process.returncode != 0:
                st.error("Pipeline failed. Check the log below.")
            else:
                st.success("Generation Complete!")
                st.session_state['current_view_path'] = output_dir
                st.rerun() # Refresh to update the Results View below

        except Exception as e:
            st.error(f"Error running pipeline: {e}")
        
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# --- 4. PERSISTENT LOG DISPLAY ---
# This section renders independent of the button, so it stays visible after rerun
if st.session_state['generation_log']:
    with st.expander("📜 Last Generation Log", expanded=False):
        st.text_area("Log Output:", value=st.session_state['generation_log'], height=300)

st.markdown("---")

# --- 5. RESULT VIEWER (Selector & Display) ---
st.subheader("3. Results Viewer")

# A. Get Run History
def get_run_folders():
    if not os.path.exists(OUTPUT_ROOT): return []
    dirs = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]
    return sorted(dirs, reverse=True)

run_folders = get_run_folders()
run_options = ["Select a run..."] + run_folders

# B. Determine Logic for Selector Index
# We try to match the currently active view path to the dropdown list
current_index = 0
if st.session_state['current_view_path']:
    current_folder_name = os.path.basename(st.session_state['current_view_path'])
    if current_folder_name in run_options:
        current_index = run_options.index(current_folder_name)

# C. Selector (Main Content Area)
col_sel, col_dummy = st.columns([1, 2]) # Use columns to keep it compact
with col_sel:
    selected_run_name = st.selectbox(
        "Choose a run to inspect:",
        options=run_options,
        index=current_index
    )

# Update the actual path based on the user's manual selection
if selected_run_name != "Select a run...":
    view_path = os.path.join(OUTPUT_ROOT, selected_run_name)
else:
    view_path = None

# D. Display Logic
# Fix for tab didnt diplayed properly
st.markdown("""
<style>
    /* Force the tab list to wrap to multiple lines */
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap;
        gap: 10px;
    }
    
    /* Hide the sliding red underline (it looks broken when wrapping) */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
            
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make st.container(border=True) border thicker */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-width: 20px;          /* bolder border */
    border-radius: 10px;        /* optional: round corners a bit more */
}
</style>
""", unsafe_allow_html=True)



if view_path and os.path.exists(view_path):
    
    # --- VIEW SETTINGS ---
    with st.expander("⚙️ View Settings", expanded=True):
        col_sort, col_filter = st.columns(2)
        with col_sort:
            sort_mode = st.radio(
                "Sort Images By:",
                ["Variation ID (Default)", "Highest Total Score", "Lowest Penalty"],
                horizontal=True
            )
        with col_filter:
            show_best_only = st.checkbox("🏆 Show Only Best Candidate per Panel")

    # Find images
    search_path = os.path.join(view_path, "*", "images", "panel*", "*_onlyname.png")
    image_files = glob.glob(search_path)
    
    panel_groups = defaultdict(list)
    panel_pattern = re.compile(r"(panel\d+)")

    for img_path in image_files:
        match = panel_pattern.search(img_path)
        if match:
            panel_name = match.group(1)
            panel_groups[panel_name].append(img_path)

    if not panel_groups:
        st.warning(f"No images found in `{selected_run_name}`.")
    else:
        # --- DISPLAY TABS ---
        panel_names = sorted(panel_groups.keys())
        tabs = st.tabs(panel_names)
        for i, panel_name in enumerate(panel_names):
            with tabs[i]:
                image_list = panel_groups[panel_name]
                
                # 1. Load Scores
                scores_map = {} 
                if image_list:
                    panel_img_dir = os.path.dirname(image_list[0])
                    score_file = os.path.join(panel_img_dir, "scores.json")
                    
                    if os.path.exists(score_file):
                        try:
                            with open(score_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                for var in data.get("variations", []):
                                    v_id = var.get("variation_id")
                                    clip = var.get("clip_score", 0.0)
                                    for layout in var.get("layout_options", []):
                                        rank = layout.get("rank")
                                        scores_map[(v_id, rank)] = {
                                            "final": layout.get("final_score", -999),
                                            "clip": clip,
                                            "sim": layout.get("sim_score", 0),
                                            "geom": layout.get("geom_penalty", 999)
                                        }
                        except Exception: pass

                # 2. Prepare Data Objects
                display_items = []
                for img_path in image_list:
                    try:
                        fname = os.path.basename(img_path)
                        parts = fname.split("_")
                        var_id = int(parts[0])
                        rank_id = int(parts[2])
                        
                        s = scores_map.get((var_id, rank_id), {
                            "final": -999, "clip": 0, "sim": 0, "geom": 0
                        })
                        
                        display_items.append({
                            "path": img_path,
                            "name": fname,
                            "score": s
                        })
                    except:
                        continue

                # 3. Sort
                if sort_mode == "Highest Total Score":
                    display_items.sort(key=lambda x: x["score"]["final"], reverse=True)
                elif sort_mode == "Lowest Penalty":
                    display_items.sort(key=lambda x: x["score"]["geom"], reverse=False)
                else:
                    display_items.sort(key=lambda x: x["name"])

                # 4. Filter
                if show_best_only and display_items:
                    if sort_mode == "Lowest Penalty":
                         best = min(display_items, key=lambda x: x["score"]["geom"])
                         display_items = [best]
                    else:
                         best = max(display_items, key=lambda x: x["score"]["final"])
                         display_items = [best]

                # 5. Render
                if not display_items:
                    st.info("No images found.")
                else:
                    cols = st.columns(3)
                    for j, item in enumerate(display_items):
                        col = cols[j % 3]
                        s = item["score"]
                        
                        try:
                            col.image(item["path"], caption=item["name"], width='stretch')
                            
                            if s["final"] != -999:
                                score_color = "green" if s['final'] > 30 else "orange" if s['final'] > 15 else "red"
                                border_color = "#d4edda" if s['final'] > 30 else "#fff3cd" if s['final'] > 15 else "#f8d7da"
                                
                                col.markdown(f"""
                                <div style="text-align:center; background-color:{border_color}; padding:8px; border-radius:8px; margin-bottom:15px; border:1px solid {score_color};">
                                    <div style="font-size:20px; font-weight:bold; color:{score_color};">
                                        {s['final']:.1f}
                                    </div>
                                    <div style="font-size:11px; color:#444; margin-top:4px;">
                                        CLIP: <b>{s['clip']:.2f}</b> • Sim: <b>{s['sim']:.2f}</b>
                                    </div>
                                    <div style="font-size:11px; color:#d9534f; font-weight:bold;">
                                        Penalty: -{s['geom']:.1f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            col.error(f"Error: {e}")