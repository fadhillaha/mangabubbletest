import math

class FixedLayoutEngine:
    def __init__(self, page_width=1000, page_height=1414, margin=80, gutter=10):
        self.w = page_width
        self.h = page_height
        self.margin = margin  # <--- NEW: Outer Safe Area
        self.gutter = gutter  # Space between panels

    def layout_page(self, panels):
        """
        Forces panels into a fixed 2x2 grid with standard Manga Margins.
        """
        if not panels:
            return []

        final_panels = []
        
        # 1. Calculate the "Live Area" (The box where panels are allowed)
        live_w = self.w - (self.margin * 2)
        live_h = self.h - (self.margin * 2)
        
        # 2. Grid Dimensions (Divides the LIVE area, not the full page)
        col_w = live_w / 2
        row_h = live_h / 2
        
        for i, p in enumerate(panels):
            new_p = p.copy()
            
            # Determine Grid Position
            # 0: Top-Right, 1: Top-Left, 2: Bottom-Right, 3: Bottom-Left
            row = 0 if i < 2 else 1
            is_right_col = (i % 2 == 0) # Even indices go to the Right (Manga RTL)
            
            # Calculate Base X/Y offsets
            if is_right_col:
                # Starts at margin + 1 column width
                x_base = self.margin + col_w
            else:
                # Starts at margin
                x_base = self.margin
                
            y_base = self.margin + (row * row_h)
            
            # 3. Create Bounding Box (Apply Gutter inside the grid cell)
            # We shrink the box by 'gutter' so panels don't touch each other
            new_p['bbox'] = [
                int(x_base + self.gutter),
                int(y_base + self.gutter),
                int(col_w - (self.gutter * 2)),
                int(row_h - (self.gutter * 2))
            ]
            
            final_panels.append(new_p)
            
        return final_panels