import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import time
from PIL import Image
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
    calculate_geometric_penalty,
    run_panel_scoring
)
from lib.script.analyze import analyze_storyboard

from lib.page.cao_initial_layout import CaoInitialLayout
from lib.page.optimizer import LayoutOptimizer
from lib.page.composite_page import PageCompositor
from lib.image.resolution import get_optimal_resolution

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline script for processing")
    parser.add_argument("--script_path", help="Path to the script file")
    parser.add_argument("--output_path", help="Path for output")
    parser.add_argument("--resume_latest", action="store_true", help="Debug mode")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--num_names", type=int, default=5)
    args = parser.parse_args()
    return args

def get_best_image_path(panel_dir):
    """Reads scores.json in the panel directory to find the winning image."""
    score_file = os.path.join(panel_dir, "scores.json")
    if not os.path.exists(score_file):
        return os.path.join(panel_dir, "00_anime.png")
        
    try:
        with open(score_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        best_path = None
        highest_score = -float('inf')
        
        for var in data.get("variations", []):
            # We want the image with the bubbles ("generated_image_path"), not the raw one
            for opt in var.get("layout_options", []):
                score = opt.get("final_score", -9999)
                if score > highest_score:
                    highest_score = score
                    # Usually: ".../02_name_0.png"
                    best_path = opt.get('generated_image_path')[:-4] + "_onlyname.png"
                    
        return best_path
    except Exception as e:
        print(f"Error reading scores for {panel_dir}: {e}")
        return os.path.join(panel_dir, "00_anime.png")

def main():
    args = parse_args()
    NUM_REFERENCES = args.num_names
    resume_latest = args.resume_latest
    # --- CONFIGURATION: MANGA PAGE ---
    # Standard B5 at roughly 150 DPI (Good for screen/web)
    PAGE_WIDTH = 1039 
    PAGE_HEIGHT = 1476
    
    # Margin (White space edge): 80px
    MARGIN = 80
    
    # Gutter (Space between panels): 15px (Increased from 10)
    GUTTER = 15 
    
    # CALCULATE LIVE AREA (Where panels go)
    LIVE_WIDTH = PAGE_WIDTH - (MARGIN * 2)
    LIVE_HEIGHT = PAGE_HEIGHT - (MARGIN * 2)
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
    print("Generating image prompts...")
    prompts = generate_image_prompts(client, panels, speakers, base_dir)
    print("Enhancing prompts...")
    prompts = enhance_prompts(client, prompts, base_dir)
    print("Calculating Page Layouts...")
    style_path = "layoutpreparation/style_models_manga109.json"
    topo_engine = CaoInitialLayout(style_path, page_width=LIVE_WIDTH, page_height=LIVE_HEIGHT, direction='rtl')
    opt_engine = LayoutOptimizer(style_path, page_width=LIVE_WIDTH, page_height=LIVE_HEIGHT, gutter=GUTTER)
    pages = {}
    for p in storyboard_data: # stored from analyze_storyboard
        p_idx = p.get('page_index', 1)
        if p_idx not in pages: pages[p_idx] = []
        pages[p_idx].append(p)

    # 2. Store resolution for every panel (panel_index -> (w, h))
    panel_resolutions = {}
    all_page_layouts = {}
    
    for page_num, page_panels in pages.items():
        # Run Cao Engine
        tree = topo_engine.generate_layout(page_panels, return_tree=True)
        final_layout = opt_engine.optimize(tree, page_panels)
        all_page_layouts[page_num] = final_layout
        
        for p in final_layout:
            # Calculate bbox from polygon
            xs = [pt[0] for pt in p['polygon']]
            ys = [pt[1] for pt in p['polygon']]
            w_raw = max(xs) - min(xs)
            h_raw = max(ys) - min(ys)
            
            # Get Safe SD Resolution
            safe_w, safe_h = get_optimal_resolution(w_raw, h_raw)
            panel_resolutions[p['panel_index']] = (safe_w, safe_h)
            # --- DEBUG PRINT ---
            raw_aspect = w_raw / h_raw if h_raw > 0 else 0
            safe_aspect = safe_w / safe_h
            print(f"  [Panel {p['panel_index']}] Layout: {int(w_raw)}x{int(h_raw)} (AR: {raw_aspect:.2f}) -> SD: {safe_w}x{safe_h} (AR: {safe_aspect:.2f})")

    num_images = args.num_images
    for i, prompt in tqdm(enumerate(prompts), desc="Generating names"):
        panel_dir = os.path.join(image_base_dir, f"panel{i:03d}")
        os.makedirs(panel_dir, exist_ok=True)
        
        sd_w, sd_h = panel_resolutions.get(i, (512, 512)) 
        print(f"  > Panel {i} Target Size: {sd_w}x{sd_h}")

        panel_entry = {
            "panel_index": i,
            "panel_dir": panel_dir,
            "prompt": prompt, # Needed for CLIP later
            "width":sd_w,
            "height":sd_h,
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
                generate_image_with_sd(prompt, image_path, width=sd_w, height=sd_h)
                if not os.path.exists(image_path):
                    continue
                anime_image_path = os.path.join(panel_dir, f"{j:02d}_anime.png")
                generate_animepose_image(image_path, prompt, anime_image_path, width=sd_w, height=sd_h)
                openpose_result = run_controlnet_openpose(image_path, anime_image_path)
                openpose_result2 = run_controlnet_openpose(anime_image_path)
                width, height = openpose_result.canvas_width, openpose_result.canvas_height
                bboxes = controlnet2bboxes(openpose_result)
                layout = generate_layout(bboxes, panels[i], sd_w, sd_h)
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

    # ==========================================
    # PAGE ASSEMBLY & PDF GENERATION
    # ==========================================
    print("\n=== Assembling Final Pages ===")
    
    # Setup Output Folder
    final_chapter_dir = os.path.join(base_dir, "final_chapter")
    os.makedirs(final_chapter_dir, exist_ok=True)
    
    # Initialize Compositor
    compositor = PageCompositor(
        page_width=PAGE_WIDTH, 
        page_height=PAGE_HEIGHT, 
        margin_x=MARGIN, 
        margin_y=MARGIN
    )
    pdf_pages = []

    for page_num, page_panels in sorted(pages.items()):
        print(f"  > Assembling Page {page_num}...")
        
        # REUSE THE SAVED LAYOUT (Do not re-generate!)
        if page_num in all_page_layouts:
            final_layout = all_page_layouts[page_num]
        else:
            print(f"    ! Error: Missing layout for Page {page_num}")
            continue
        
        # B. Collect Winning Images
        page_image_map = {}
        for p in final_layout:
            idx = p['panel_index']
            panel_dir = os.path.join(image_base_dir, f"panel{idx:03d}")
            
            # Find the best image using the helper
            best_img = get_best_image_path(panel_dir)
            
            if best_img and os.path.exists(best_img):
                page_image_map[idx] = best_img
            else:
                page_image_map[idx] = os.path.join(panel_dir,"00_anime.png")
                print(f"    ! Warning: No valid image found for Panel {idx}")

        # C. Composite Page
        output_filename = f"page_{page_num:02d}.png"
        output_path = os.path.join(final_chapter_dir, output_filename)
        compositor.create_page(final_layout, page_image_map, output_path)
        
        # D. Add to PDF List
        if os.path.exists(output_path):
            try:
                # Open and convert to RGB (drop alpha) for PDF compatibility
                img_obj = Image.open(output_path).convert("RGB")
                pdf_pages.append(img_obj)
            except Exception as e:
                print(f"    ! Error loading page for PDF: {e}")

    # 6. Save PDF
    if pdf_pages:
        pdf_path = os.path.join(final_chapter_dir, "manga_chapter.pdf")
        print(f"\n📚 Saving Full Chapter PDF: {pdf_path}")
        pdf_pages[0].save(
            pdf_path, 
            save_all=True, 
            append_images=pdf_pages[1:]
        )
        print("✅ Pipeline Finished Successfully!")
    else:
        print("⚠️ No pages were generated.")

if __name__ == "__main__":
    main()