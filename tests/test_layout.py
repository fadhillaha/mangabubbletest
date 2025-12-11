import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from lib.page.cao_initial_layout import CaoInitialLayout
from lib.page.optimizer import LayoutOptimizer

# 1. Setup paths
MODEL_PATH = "layoutpreparation/style_models.json" 
METADATA_PATH = "output\\ui_run_20251203_151436\\20251203_1515\\panel_metadata.json"

# Load Data
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_panels = json.load(f)
page_1 = [p for p in all_panels if p.get('page_index') == 1]

# Initialize Engines
initial_engine = CaoInitialLayout(style_model_path=MODEL_PATH, direction='rtl')
optimizer = LayoutOptimizer(style_model_path=MODEL_PATH, gutter=30)

# Generate Top 5 Trees
print("Stage 1: Generating Top 5 Candidate Trees...")
top_results = initial_engine.generate_top_k(page_1, k=5, return_trees=True)

# --- VISUALIZATION ---
# Create 2 Rows x 5 Columns
fig, axes = plt.subplots(2, 5, figsize=(25, 12))
fig.suptitle(f"Layout Optimization: Before (Stage 1) vs After (Stage 2)", fontsize=16)

print(f"Visualizing {len(top_results)} layouts...")

for i, (score, tree) in enumerate(top_results):
    
    # --- ROW 1: BEFORE (Stage 1 - Rectangles) ---
    ax_before = axes[0, i]
    ax_before.set_xlim(0, 1000)
    ax_before.set_ylim(1414, 0)
    ax_before.set_aspect('equal')
    ax_before.set_title(f"Rank {i+1} (Stage 1)\nRaw Topology", fontsize=10)
    
    # Use the internal helper to see the flat rectangles from the tree
    raw_layout = initial_engine._flatten_tree(tree, page_1)
    
    for p in raw_layout:
        x, y, w, h = p['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='#ffeeee')
        ax_before.add_patch(rect)
        ax_before.text(x + w/2, y + h/2, f"P{p['panel_index']}", ha='center', color='red')

    # --- ROW 2: AFTER (Stage 2 - Optimized Polygons) ---
    ax_after = axes[1, i]
    ax_after.set_xlim(0, 1000)
    ax_after.set_ylim(1414, 0)
    ax_after.set_aspect('equal')
    ax_after.set_title(f"Rank {i+1} (Stage 2)\nOptimized & Guttered", fontsize=10)
    
    # Run Optimizer
    final_layout = optimizer.optimize(tree, page_1)
    
    for p in final_layout:
        poly = p['polygon']
        poly_closed = poly + [poly[0]] 
        
        xs, ys = zip(*poly_closed)
        ax_after.plot(xs, ys, linewidth=2, color='black')
        ax_after.fill(xs, ys, alpha=0.1, color='blue')
        
        # Centroid for label
        cx = sum(x for x,y in poly)/len(poly)
        cy = sum(y for x,y in poly)/len(poly)
        ax_after.text(cx, cy, f"P{p['panel_index']}", ha='center')

plt.tight_layout()
plt.show()