import os
import sys
import json
import glob
import requests
import base64
import io
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.scoring.scorer import GeometricScorer
from lib.image.controlnet import run_controlnet_openpose, controlnet2bboxes
from lib.layout.layout import generate_layout, similar_layouts

def test_scoring():
    # 1. Find Latest Run
    output_root = "output\\ui_run_20251119_125517"
    if not os.path.exists(output_root):
        print("No output directory found.")
        return

    # Filter for directories only
    runs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    runs.sort(reverse=True)
    
    if not runs:
        print("No runs found.")
        return
    
    target_run = os.path.join(output_root, runs[0]) # Use latest run
    print(f"🔍 Analyzing Run: {target_run}")

    # 2. Load Panel Data (Dialogue)
    # Try finding the script JSON in different possible locations
    panel_json_path = os.path.join(target_run, "script", "panel.json")
    if not os.path.exists(panel_json_path):
        panel_json_path = os.path.join(target_run, "panel.json")
    
    if not os.path.exists(panel_json_path):
        print("❌ Could not find panel.json. Cannot reconstruct layout logic.")
        return
        
    with open(panel_json_path, "r", encoding="utf-8") as f:
        panels = json.load(f)

    # 3. Initialize Scorer
    geo_scorer = GeometricScorer()
    
    images_root = os.path.join(target_run, "images")
    if not os.path.exists(images_root):
        print("❌ No images folder found.")
        return

    # 4. Loop through Panels
    for i, panel_data in enumerate(panels):
        panel_name = f"panel{i:03d}"
        panel_dir = os.path.join(images_root, panel_name)
        
        if not os.path.exists(panel_dir):
            continue
            
        print(f"\n--- Analyzing {panel_name} ---")
        
        # Find generated images (00.png, 01.png...)
        # We exclude filenames containing "anime", "openpose", "name" to get just the base images
        all_files = sorted(glob.glob(os.path.join(panel_dir, "*.png")))
        base_images = [f for f in all_files if "anime" not in f and "openpose" not in f and "name" not in f]
        
        candidates = []

        for img_path in base_images:
            filename = os.path.basename(img_path)
            print(f"  Processing {filename}...")
            
            # A. Re-Run OpenPose (Crucial for Face Detection)
            # We check if a sketch exists to use as guidance, otherwise use main image
            sketch_path = img_path.replace(".png", "_anime.png")
            if not os.path.exists(sketch_path): 
                sketch_path = img_path
            
            try:
                # Call Local WebUI API
                openpose_result = run_controlnet_openpose(img_path, sketch_path)
                
                # B. Re-Calculate Layout
                bboxes = controlnet2bboxes(openpose_result)
                width, height = openpose_result.canvas_width, openpose_result.canvas_height
                layout = generate_layout(bboxes, panel_data, width, height)
                
                if layout is None:
                    print("    [Skipping] Invalid Layout (Character mismatch)")
                    continue
                    
                scored_layouts = similar_layouts(layout)
                if not scored_layouts:
                    print("    [Skipping] No database match")
                    continue
                
                best_template = scored_layouts[0]

                # C. CALCULATE GEOMETRIC SCORE
                penalty = geo_scorer.score(openpose_result, best_template, panel_data)
                
                candidates.append({
                    "file": filename,
                    "penalty": penalty
                })
                
                print(f"    -> Penalty: {penalty:.2f}")
                
            except Exception as e:
                print(f"    [Error] {e}")

        # Pick Winner
        if candidates:
            # Sort by Penalty (Lowest is Best)
            candidates.sort(key=lambda x: x["penalty"])
            winner = candidates[0]
            print(f"  🏆 WINNER: {winner['file']} (Penalty: {winner['penalty']:.2f})")
        else:
            print("  ⚠️ No valid candidates found.")

if __name__ == "__main__":
    test_scoring()