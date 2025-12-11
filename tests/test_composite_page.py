import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lib.page.cao_initial_layout import CaoInitialLayout
from lib.page.optimizer import LayoutOptimizer
from lib.page.composite_page import PageCompositor

# --- CONFIG ---
MODEL_PATH = "layoutpreparation/style_models.json"
RUN_DIR = "output/ui_run_20251205_155059/20251205_1551" # <--- UPDATE THIS

def get_best_image_for_panel(run_dir, panel_idx):
    """Parses scores.json to find the image path of the highest scoring variation."""
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
                    best_image_path = opt.get('generated_image_path')[:-4] + "_onlyname.png"
                    

        if best_image_path:
            # Fix absolute/relative path issues
            if not os.path.exists(best_image_path):
                # Try relative to the script location if absolute fails
                rel_path = os.path.abspath(best_image_path)
                if os.path.exists(rel_path):
                    best_image_path = rel_path
            return best_image_path

    except Exception as e:
        print(f"  [Error] Failed parsing scores for Panel {panel_idx}: {e}")
        return None
    return None

def main():
    if not os.path.exists(RUN_DIR):
        print(f"❌ Error: RUN_DIR not found: {RUN_DIR}")
        return

    metadata_path = os.path.join(RUN_DIR, "panel_metadata.json")
    if not os.path.exists(metadata_path):
        print("❌ Error: panel_metadata.json not found.")
        return

    # 1. Setup Output Folder
    final_output_dir = os.path.join(RUN_DIR, "final_chapter")
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📂 Output Folder created: {final_output_dir}")

    # 2. Load Metadata & Group
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_panels = json.load(f)

    pages = {}
    for p in all_panels:
        p_idx = p.get('page_index', 1)
        if p_idx not in pages: pages[p_idx] = []
        pages[p_idx].append(p)

    # --- CONFIGURATION: MANGA PAGE (B5) ---
    PAGE_WIDTH = 1039   # B5 Width (approx 150dpi)
    PAGE_HEIGHT = 1476  # B5 Height
    MARGIN = 80         # White border size
    GUTTER = 15         # Space between panels
    
    # Calculate Live Area (Where panels go)
    LIVE_WIDTH = PAGE_WIDTH - (MARGIN * 2)
    LIVE_HEIGHT = PAGE_HEIGHT - (MARGIN * 2)
    
    print(f"⚙️ Config: B5 Page ({PAGE_WIDTH}x{PAGE_HEIGHT}) | Live Area ({LIVE_WIDTH}x{LIVE_HEIGHT})")

    # 3. Initialize Engines
    # Layout Engines work on the LIVE AREA (Internal Box)
    topo_engine = CaoInitialLayout(
        MODEL_PATH, 
        page_width=LIVE_WIDTH, 
        page_height=LIVE_HEIGHT, 
        direction='rtl'
    )
    
    opt_engine = LayoutOptimizer(
        MODEL_PATH, 
        page_width=LIVE_WIDTH, 
        page_height=LIVE_HEIGHT,
        gutter=10
    )
    
    # Compositor works on the FULL PAGE (Paper Size)
    compositor = PageCompositor(
        page_width=PAGE_WIDTH, 
        page_height=PAGE_HEIGHT, 
        margin_x=MARGIN, 
        margin_y=MARGIN
    )

    pdf_pages = []

    # 4. Process Each Page
    for page_num, panel_list in sorted(pages.items()):
        print(f"\n=== 📄 Processing Page {page_num} ===")
        
        # A. Layout
        tree = topo_engine.generate_layout(panel_list, return_tree=True)
        final_layout = opt_engine.optimize(tree, panel_list)
        
        print(f"  > Layout Geometry for Page {page_num}:")
        for p in final_layout:
            idx = p['panel_index']
            poly = p['polygon']
            # Calculate Bounds
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            ratio = w / h if h > 0 else 0
            print(f"    - Panel {idx}: {int(w)}x{int(h)} px (Ratio: {ratio:.2f})")
        
        # B. Image Selection
        image_map = {}
        for p in final_layout:
            idx = p['panel_index']
            img_path = get_best_image_for_panel(RUN_DIR, idx)
            if img_path: image_map[idx] = img_path

        # C. Composite & Save
        output_filename = f"page_{page_num:02d}.png"
        output_path = os.path.join(final_output_dir, output_filename)
        
        compositor.create_page(final_layout, image_map, output_path)
        
        # D. Add to PDF
        if os.path.exists(output_path):
            img_obj = Image.open(output_path).convert("RGB")
            pdf_pages.append(img_obj)

    # 5. Save PDF
    if pdf_pages:
        pdf_path = os.path.join(final_output_dir, "manga_chapter.pdf")
        print(f"\n📚 Saving PDF to: {pdf_path}")
        pdf_pages[0].save(
            pdf_path, 
            save_all=True, 
            append_images=pdf_pages[1:]
        )
        print("✅ Done!")
    else:
        print("⚠️ No pages were generated.")

if __name__ == "__main__":
    main()