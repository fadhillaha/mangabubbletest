import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import time
#from openai import OpenAI
from lib.llm.geminiadapter import GeminiClient
from dotenv import load_dotenv
from lib.layout.layout import generate_layout, similar_layouts
from lib.script.divide import divide_script, ele2panels, refine_elements
from lib.image.image import (
    generate_image,
    generate_image_prompts,
    enhance_prompts,
    generate_image_with_sd,
)
from lib.image.controlnet import (
    detect_human,
    check_open,
    controlnet2bboxes,
    run_controlnet_openpose,
)
from lib.name.name import generate_name, generate_animepose_image

from lib.scoring.scorer import (
    load_clip_model,
    get_verification_prompt,
    calculate_clip_score,
    calculate_geometric_penalty,
    run_panel_scoring
)
from lib.script.analyze import analyze_storyboard

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline script for processing")
    parser.add_argument("--script_path", help="Path to the script file")
    parser.add_argument("--output_path", help="Path for output")
    parser.add_argument("--resume_latest", action="store_true", help="Debug mode")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--num_names", type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    NUM_REFERENCES = args.num_names
    resume_latest = args.resume_latest
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or OPENAI_API_KEY is not set")
    if not check_open():
        raise ValueError("ControlNet is not running")
    if resume_latest:
        date_dirs = [
            d
            for d in os.listdir(args.output_path)
            if os.path.isdir(os.path.join(args.output_path, d))
        ]
        date_dirs.sort(key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M"))
        date = date_dirs[-1]
        base_dir = os.path.join(args.output_path, date)
        image_base_dir = os.path.join(base_dir, "images")
    else:
        date = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = os.path.join(args.output_path, date)
        image_base_dir = os.path.join(base_dir, "images")
        if not os.path.exists(image_base_dir):
            os.makedirs(image_base_dir)

    #client = OpenAI()
    client = GeminiClient(api_key=api_key, model="gemini-2.5-flash")
    print("Dividing script...")
    elements = divide_script(client, args.script_path, base_dir)
    print("Refining elements...")
    elements = refine_elements(elements, base_dir)
    print("Detected Speakers:")
    speakers = set()
    for element in elements:
        speakers.add(element["speaker"])
    speakers = list(speakers - {""})
    print(speakers)
    print("Converting elements to panels...")
    panels = ele2panels(client, elements, base_dir)
    print("Generating Storyboard Metadata...")
    storyboard_data = analyze_storyboard(client, panels, base_dir)
    if storyboard_data:
        first = storyboard_data[0]
        print(f" > First Panel Strategy: {first.get('suggested_aspect_ratio', 'N/A')} ({first.get('reasoning', '')})")
    # -------------------------
    print("Generating image prompts...")
    prompts = generate_image_prompts(client, panels, speakers, base_dir)
    print("Enhancing prompts...")
    prompts = enhance_prompts(client, prompts, base_dir)
    num_images = args.num_images
    for i, prompt in tqdm(enumerate(prompts), desc="Generating names"):
        panel_dir = os.path.join(image_base_dir, f"panel{i:03d}")
        os.makedirs(panel_dir, exist_ok=True)
        
        panel_entry = {
            "panel_index": i,
            "panel_dir": panel_dir,
            "prompt": prompt, # Needed for CLIP later
            "variations": []
        }

        for j in range(num_images):

            max_retries = 3  
            layout = None
            openpose_result = None

            for attempt in range(max_retries):
                print(f"  > Generating image {j} (Attempt {attempt+1}/{max_retries})...")
                image_path = os.path.join(panel_dir, f"{j:02d}.png")
                # generate_image(client, prompt, image_path)
                generate_image_with_sd(prompt, image_path)
                if not os.path.exists(image_path):
                    continue
                anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
                generate_animepose_image(image_path, prompt, anime_image_path)
                openpose_result = run_controlnet_openpose(image_path, anime_image_path)
                openpose_result2 = run_controlnet_openpose(anime_image_path)
                width, height = openpose_result.canvas_width, openpose_result.canvas_height
                bboxes = controlnet2bboxes(openpose_result)
                layout = generate_layout(bboxes, panels[i], width, height)
                if layout is not None:
                    break # found a valid layout
                else:
                    print("    ! Invalid layout detected, retrying...")
            if layout is None:
                print(f"    ! Failed to generate a valid layout after {max_retries} attempts. Skipping this image.")
                continue
            
    
            scored_layouts = similar_layouts(layout)
            layout_options = []
            for idx_ref_layout, scored_layout in enumerate(
                scored_layouts[:NUM_REFERENCES]
            ):
                print(scored_layout[0].image_path)
                save_path = os.path.join(
                    panel_dir, f"{j:02d}_name_{idx_ref_layout:1d}.png"
                )
                ref_layout = scored_layout[0] # The template object
                sim_score = scored_layout[1]
                geom_penalty = calculate_geometric_penalty(
                        ref_layout, 
                        panels[i], 
                        openpose_result2,
                        os.path.join(
                    panel_dir, f"{j:02d}_name_{idx_ref_layout:1d}_bbox.png"
                )
                    )


                generate_name(
                    openpose_result, layout, scored_layout, panels[i], save_path
                )

                layout_options.append({
                        "rank": idx_ref_layout,
                        "template_path": ref_layout.image_path,
                        "generated_image_path": save_path, # Saved for reference
                        "sim_score": sim_score,
                        "geom_penalty": geom_penalty,
                    })
                
            panel_entry["variations"].append({
                    "variation_id": j,
                    "image_path": image_path,        # Raw image (for CLIP scoring)
                    "anime_image_path": anime_image_path,
                    "layout_options": layout_options # Contains the Geom Scores & Output Paths
                })
            score_file_path = os.path.join(panel_dir, "scores.json")
            with open(score_file_path, "w", encoding="utf-8") as f:
                json.dump(panel_entry, f, indent=4, default=str)

    # ==========================================
    # SCORING PART
    # ==========================================
    run_panel_scoring(base_dir, prompts)

if __name__ == "__main__":
    main()