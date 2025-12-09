import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from lib.page.cao_initial_layout import CaoInitialLayout
from lib.page.optimizer import LayoutOptimizer

# 1. Setup paths
# Use the style_models.json you just generated in layoutpreparation/
MODEL_PATH = "layoutpreparation/style_models.json" 

# Use your existing panel metadata
METADATA_PATH = "output\\ui_run_20251203_151436\\20251203_1515\\panel_metadata.json" # UPDATE THIS

# Load Data
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_panels = json.load(f)
page_1 = [p for p in all_panels if p.get('page_index') == 1]

# --- STAGE 1: Topology ---
print("Stage 1: Generating Rectangular Tree...")
initial_engine = CaoInitialLayout(style_model_path=MODEL_PATH ,direction='ltr')
# NOTE: We need the Tree structure, so return_tree=True
rect_tree = initial_engine.generate_layout(page_1, return_tree=True) 

# --- STAGE 2: Geometry Optimization ---
print("Stage 2: Optimizing Shapes (Creating Diagonals)...")
optimizer = LayoutOptimizer(style_model_path=MODEL_PATH, gutter=30 )

# FIX: Pass BOTH the tree and the metadata
final_layout = optimizer.optimize(rect_tree, page_1)

# --- VISUALIZE ---
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 1000)
ax.set_ylim(1414, 0)

for p in final_layout:
    poly = p['polygon'] # List of [x,y]
    poly_closed = poly + [poly[0]] # Close loop for drawing
    
    xs, ys = zip(*poly_closed)
    ax.plot(xs, ys, linewidth=2, color='black')
    ax.fill(xs, ys, alpha=0.1, color='blue')
    
    # Label center
    cx = sum(x for x,y in poly)/len(poly)
    cy = sum(y for x,y in poly)/len(poly)
    ax.text(cx, cy, f"P{p['panel_index']}", ha='center')

plt.title("Final Cao Layout (Stage 1 + Stage 2)")
plt.show()