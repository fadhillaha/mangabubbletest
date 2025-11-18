import streamlit as st
import os
import json
import sys
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# --- 0. Setup & Imports ---
try:
    # Import your Adapter Client
    from lib.llm.geminiadapter import GeminiClient
    
    # Import the original library functions (no modification needed!)
    from lib.script.divide import divide_script, ele2panels, refine_elements
    from lib.image.image import generate_image_prompts, enhance_prompts, generate_image_with_sd
    from lib.image.controlnet import run_controlnet_openpose, controlnet2bboxes, ControlNetResult
    from lib.name.name import generate_animepose_image, generate_name
    from lib.layout.layout import generate_layout, similar_layouts
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.error("Make sure you run this script from the project root and have run 'pip install -e .'")
    st.stop()

# Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set in .env file")
    st.stop()

# --- Initialize Gemini Client (Adapter) ---
# This mimics the OpenAI client structure that your lib functions expect
try:
    # Using the model version you specified
    client = GeminiClient(api_key=api_key, model="gemini-2.0-flash")
except Exception as e:
    st.error(f"Failed to initialize GeminiClient: {e}")
    st.stop()

# Config
OUTPUT_DIR = "output_debug_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_dir = OUTPUT_DIR

st.set_page_config(page_title="Full Pipeline Debugger", layout="wide")
st.title("🚀 Full Pipeline Debugger (Gemini Adapter)")

# --- 1. Script Input ---
st.sidebar.header("Input Script")
default_script = """cene: A high school classroom in Japan during late afternoon. The room is bathed in the warm, golden glow of the sunset coming through the windows. Shadows are long.
Kenji (A teenage boy with messy brown hair and a loose tie, looking anxious and leaning forward): "Hey Aiko, did you actually finish the math homework? It was impossible!"
Aiko (A teenage girl with long black hair, wearing a neat school uniform, smiling confidently while holding up a notebook): "Of course! It was easy once you figured out the formula."
Kenji (Sighing deeply, slumping over his desk in defeat, scratching his head): "I didn't understand it at all... I'm doomed."
"""
script_text = st.sidebar.text_area("Enter Script", default_script, height=300)

if st.button("Run Full Pipeline", type="primary"):
    
    # 1. Prepare Script File
    script_path = os.path.join(OUTPUT_DIR, "temp_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    st.header("1. Script Processing (LLM)")
    
    # --- Step 1: Divide Script ---
    with st.status("Parsing Script...", expanded=True) as status:
        st.write("running `divide_script`...")
        # We pass 'client' (your GeminiAdapter) instead of 'model'
        elements = divide_script(client, script_path, base_dir)
        
        st.write("running `refine_elements`...")
        elements = refine_elements(elements, base_dir)
        
        st.json(elements, expanded=False)
        
        # Extract Speakers
        speakers = list(set([e["speaker"] for e in elements if "speaker" in e]) - {""})
        st.success(f"Detected Speakers: {speakers}")

        # --- Step 2: Elements to Panels ---
        st.write("running `ele2panels`...")
        panels = ele2panels(client, elements, base_dir)
        st.json(panels, expanded=False)
        
        # --- Step 3: Generate Prompts ---
        st.write("running `generate_image_prompts`...")
        prompts = generate_image_prompts(client, panels, speakers, base_dir)
        st.write(f"Generated Content Tags: `{prompts[0]}`") 
        
        # --- Step 4: Enhance Prompts ---
        st.write("running `enhance_prompts`...")
        enhanced_prompts = enhance_prompts(client, prompts, base_dir)
        
        final_prompt = enhanced_prompts[0]
        if isinstance(final_prompt, dict):
            final_prompt = final_prompt["prompt"]
            
        st.info(f"**Final Enhanced Prompt:** {final_prompt}")
        status.update(label="Script Processing Complete!", state="complete", expanded=False)

    st.header("2. Image Generation & Analysis")
    
    # Process First Panel Only
    panel_idx = 0
    panel_data = panels[panel_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("A. Image Generation")
        with st.spinner("Generating Images with Stable Diffusion..."):
            img_name = f"panel{panel_idx}_00.png"
            img_path = os.path.join(OUTPUT_DIR, img_name)
            anime_path = os.path.join(OUTPUT_DIR, f"panel{panel_idx}_00_anime.png")
            
            # 1. Main Image
            generate_image_with_sd(final_prompt, img_path)
            st.image(img_path, caption="Main Image")
            
            # 2. Anime Sketch
            generate_animepose_image(img_path, final_prompt, anime_path)
            st.image(anime_path, caption="Sketch (for OpenPose)")

    with col2:
        st.subheader("B. ControlNet Analysis")
        with st.spinner("Running OpenPose..."):
            # 3. ControlNet
            openpose_result = run_controlnet_openpose(img_path, anime_path)
            
            if openpose_result.image:
                st.image(openpose_result.image, caption="OpenPose Skeleton")
            
            # 4. BBoxes
            bboxes = controlnet2bboxes(openpose_result)
            
            # Visualizing BBoxes
            base_img = Image.open(img_path)
            draw = ImageDraw.Draw(base_img)
            for i, box in enumerate(bboxes):
                draw.rectangle(box, outline="red", width=5)
                draw.text((box[0], box[1]), f"Char {i}", fill="red")
            st.image(base_img, caption="Detected Bounding Boxes")
            st.write(f"BBoxes Data: {bboxes}")

    st.header("3. Layout & Bubble Placement")
    
    with st.spinner("Calculating Layout..."):
        width, height = openpose_result.canvas_width, openpose_result.canvas_height
        
        # 5. Generate Layout Object
        layout = generate_layout(bboxes, panel_data, width, height)
        
        if layout is None:
            st.error("Failed to generate layout. Likely mismatch between detected characters and dialogue.")
        else:
            # 6. Similar Layouts
            scored_layouts = similar_layouts(layout)
            
            if not scored_layouts:
                st.error("No matching layouts found in database.")
            else:
                best_match = scored_layouts[0]
                ref_layout = best_match[0]
                score = best_match[1]
                pairs_iterator = best_match[2]
                
                # Convert zip object to list for display and reuse
                pairs_list = list(pairs_iterator)
                
                st.success(f"Best Match Found! Score: {score:.4f}")
                st.write(f"Template Image: `{ref_layout.image_path}`")
                st.write("### 🔗 Pairs Mapping")
                st.code(str(pairs_list), language="python")
                
                # 7. Generate Final Name
                reconstructed_scored_layout = (ref_layout, score, pairs_list)
                save_name_path = os.path.join(OUTPUT_DIR, f"panel{panel_idx}_final.png")
                
                generate_name(
                    openpose_result, 
                    layout, 
                    reconstructed_scored_layout, 
                    panel_data, 
                    save_name_path
                )
                
                st.image(save_name_path, caption="✨ Final Result with Bubbles", use_column_width=True)