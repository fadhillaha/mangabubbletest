import os
from PIL import Image, ImageDraw
import numpy as np

class PageCompositor:
    def __init__(self, page_width=1000, page_height=1414, gutter_color=(0, 0, 0)):
        self.w = page_width
        self.h = page_height
        self.bg_color = gutter_color

    def create_page(self, panels_layout, image_paths, output_path):
        """
        Stitches images onto the page based on the layout polygons.
        
        Args:
            panels_layout (list): List of dicts with 'polygon' [[x,y]...] and 'panel_index'.
            image_paths (dict): Mapping {panel_index: "path/to/image.png"}.
            output_path (str): Where to save the final page.
        """
        # 1. Create Blank Page
        page = Image.new("RGB", (self.w, self.h), self.bg_color)
        draw = ImageDraw.Draw(page)

        print(f"Compositing {len(panels_layout)} panels...")

        # 2. Iterate through panels
        for p in panels_layout:
            idx = p['panel_index']
            
            # Skip if we don't have an image for this panel
            if idx not in image_paths:
                print(f"  [Warning] No image found for Panel {idx}")
                continue

            img_path = image_paths[idx]
            if not os.path.exists(img_path):
                print(f"  [Warning] Image file missing: {img_path}")
                continue
            
            try:
                original_img = Image.open(img_path).convert("RGBA")
            except Exception as e:
                print(f"  [Error] Failed to load image {img_path}: {e}")
                continue
            
            # --- A. Prepare the Geometry ---
            poly_pts = [tuple(pt) for pt in p['polygon']]
            
            # Calculate Bounding Box of the Polygon
            xs = [pt[0] for pt in poly_pts]
            ys = [pt[1] for pt in poly_pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Width/Height of the target area
            target_w = int(max_x - min_x)
            target_h = int(max_y - min_y)
            
            if target_w <= 0 or target_h <= 0: continue

            # --- B. Create the Mask ---
            # We work in a "local" buffer size of the bounding box to save memory
            mask = Image.new("L", (target_w, target_h), 0)
            mask_draw = ImageDraw.Draw(mask)
            
            # Shift polygon points to be relative to the top-left of the mask
            local_poly = [(pt[0] - min_x, pt[1] - min_y) for pt in poly_pts]
            mask_draw.polygon(local_poly, fill=255)

            # --- C. Aspect Fit the Image ---
            # We need to resize the source image so it covers the mask entirely
            # avoiding any empty space (Aspect Fill / "Cover" mode)
            img_ratio = original_img.width / original_img.height
            target_ratio = target_w / target_h
            
            if img_ratio > target_ratio:
                # Image is wider than target -> Match Height, Crop Width
                new_h = target_h
                new_w = int(target_h * img_ratio)
            else:
                # Image is taller than target -> Match Width, Crop Height
                new_w = target_w
                new_h = int(target_w / img_ratio)
                
            # High-quality resize
            resized_img = original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center crop
            crop_x = (new_w - target_w) // 2
            crop_y = (new_h - target_h) // 2
            cropped_img = resized_img.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))

            # --- D. Paste onto Page ---
            # Paste using the mask
            page.paste(cropped_img, (int(min_x), int(min_y)), mask)
            
            # --- E. Draw Borders ---
            # Draw a thick black line around the polygon to hide jagged edges
            draw.line(poly_pts + [poly_pts[0]], fill=(0, 0, 0), width=5)

        # 3. Save Final Result
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        page.save(output_path)
        print(f"✅ Page saved to: {output_path}")
        return output_path