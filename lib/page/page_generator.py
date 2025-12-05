import os
import json
import numpy as np
from PIL import Image
from lib.page.cao_initial_layout import CaoInitialLayout
from lib.page.optimizer import LayoutOptimizer
from lib.page.composite_page import PageCompositor

def get_best_image_for_panel(run_dir, panel_idx):
    """
    Parses scores.json to find the image path of the highest scoring variation.
    """
    panel_dir = os.path.join(run_dir, "images", f"panel{panel_idx:03d}")
    score_file = os.path.join(panel_dir, "scores.json")
    
    if not os.path.exists(score_file):
        print(f"  [Warn] Score file missing for Panel {panel_idx}")
        return None

    try:
        with open(score_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        best_image_path = None
        highest_score = -float('inf')
        
        variations = data.get("variations", [])
        if not variations: return None

        for var in variations:
            options = var.get("layout_options", [])
            for opt in options:
                score = opt.get("final_score", -9999)
                if score > highest_score:
                    highest_score = score
                    # Construct correct path to the named image
                    raw_path = opt.get('generated_image_path')
                    if raw_path:
                        # Ensure we get the name-composited version
                        best_image_path = raw_path
                    
        if best_image_path:
            if not os.path.exists(best_image_path):
                # Try correcting relative path
                rel_path = os.path.abspath(best_image_path)
                if os.path.exists(rel_path):
                    best_image_path = rel_path
            return best_image_path

    except Exception as e:
        print(f"  [Error] Failed parsing scores for Panel {panel_idx}: {e}")
        return None
    return None

def create_manga_chapter(run_dir, model_path, output_folder="final_chapter", direction='rtl'):
    """
    Main function to assemble a manga chapter from generated panels.
    
    Args:
        run_dir (str): Path to the pipeline output directory.
        model_path (str): Path to style_models.json.
        output_folder (str): Subfolder name for output.
        direction (str): 'rtl' (Manga) or 'ltr' (Western).
    """
    if not os.path.exists(run_dir):
        print(f"❌ Error: RUN_DIR not found: {run_dir}")
        return

    metadata_path = os.path.join(run_dir, "panel_metadata.json")
    if not os.path.exists(metadata_path):
        print("❌ Error: panel_metadata.json not found.")
        return

    # 1. Setup Output
    final_output_dir = os.path.join(run_dir, output_folder)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📂 Output Folder: {final_output_dir}")

    # 2. Load Metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_panels = json.load(f)

    pages = {}
    for p in all_panels:
        p_idx = p.get('page_index', 1)
        if p_idx not in pages: pages[p_idx] = []
        pages[p_idx].append(p)

    # 3. Initialize Engines
    print(f"⚙️ Initializing Layout Engines ({direction.upper()})...")
    topo_engine = CaoInitialLayout(model_path, direction=direction)
    opt_engine = LayoutOptimizer(model_path)
    compositor = PageCompositor()

    pdf_pages = []

    # 4. Process Each Page
    for page_num, panel_list in sorted(pages.items()):
        print(f"\n=== 📄 Processing Page {page_num} ===")
        
        # A. Layout
        tree = topo_engine.generate_layout(panel_list, return_tree=True)
        final_layout = opt_engine.optimize(tree, panel_list)
        
        # B. Image Selection
        image_map = {}
        for p in final_layout:
            idx = p['panel_index']
            img_path = get_best_image_for_panel(run_dir, idx)
            if img_path: 
                image_map[idx] = img_path
            else:
                print(f"    ⚠️ Warning: No image found for Panel {idx}")

        # C. Composite
        output_filename = f"page_{page_num:02d}.png"
        output_path = os.path.join(final_output_dir, output_filename)
        
        compositor.create_page(final_layout, image_map, output_path)
        
        if os.path.exists(output_path):
            try:
                img_obj = Image.open(output_path).convert("RGB")
                pdf_pages.append(img_obj)
            except Exception as e:
                print(f"    ❌ Failed to load page image: {e}")

    # 5. Save PDF
    if pdf_pages:
        pdf_path = os.path.join(final_output_dir, "manga_chapter.pdf")
        print(f"\n📚 Saving PDF to: {pdf_path}")
        pdf_pages[0].save(
            pdf_path, 
            save_all=True, 
            append_images=pdf_pages[1:]
        )
        print("✅ Chapter Generation Complete!")
    else:
        print("⚠️ No pages generated.")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Example Usage
    MODEL_PATH = "layoutpreparation/style_models.json"
    RUN_DIR = "output/ui_run_20251203_151436/20251203_1515" 
    
    create_manga_chapter(RUN_DIR, MODEL_PATH, direction='rtl')