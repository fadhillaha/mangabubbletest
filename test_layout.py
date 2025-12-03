import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lib.page.page_layout import PageLayoutEngine # <--- Using your new file name

# 1. Load the metadata you just generated
METADATA_PATH = "output\\ui_run_20251203_113053\\20251203_1131\\panel_metadata.json" # UPDATE THIS PATH
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_panels = json.load(f)

# 2. Filter for Page 1 only
page_1_panels = [p for p in all_panels if p.get('page_index') == 3]
print(f"Testing Layout for Page 1 ({len(page_1_panels)} panels)...")

# 3. Run the Engine
engine = PageLayoutEngine()
layout_result = engine.layout_page(page_1_panels)

# 4. Visualize
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 500)
ax.set_ylim(750, 0) # Invert Y

for p in layout_result:
    x, y, w, h = p['bbox']
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    
    # Text Label
    cx, cy = x + w/2, y + h/2
    info = f"P{p['panel_index']}\nScore:{p['importance_score']}\n{p.get('suggested_aspect_ratio','')}"
    ax.text(cx, cy, info, ha='center', va='center', fontsize=9)

plt.title("Generated Layout: Page 1")
plt.show()