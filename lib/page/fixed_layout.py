import math

class FixedLayoutEngine:
    def __init__(self, page_width=1000, page_height=1414, gutter=10):
        self.w = page_width
        self.h = page_height
        self.gutter = gutter

    def layout_page(self, panels):
        """
        Forces panels into a fixed 2x2 grid.
        Order: Top-Right -> Top-Left -> Bottom-Right -> Bottom-Left (Manga Style)
        """
        if not panels:
            return []

        final_panels = []
        
        # Grid Dimensions
        # We assume 2 columns and 2 rows
        col_w = self.w / 2
        row_h = self.h / 2
        
        for i, p in enumerate(panels):
            new_p = p.copy()
            
            # Determine Grid Position (Index 0 to 3)
            # 0: Top-Right (Col 1, Row 0)
            # 1: Top-Left  (Col 0, Row 0)
            # 2: Bottom-Right (Col 1, Row 1)
            # 3: Bottom-Left (Col 0, Row 1)
            
            row = 0 if i < 2 else 1
            
            # RTL Logic: Even index (0, 2) is Right Column. Odd (1, 3) is Left.
            is_right_col = (i % 2 == 0)
            
            if is_right_col:
                x = col_w
            else:
                x = 0
                
            y = row * row_h
            
            # Create Bounding Box with Gutter
            new_p['bbox'] = [
                int(x + self.gutter),
                int(y + self.gutter),
                int(col_w - (self.gutter * 2)),
                int(row_h - (self.gutter * 2))
            ]
            
            # Optional: Add "Polygon" for compatibility with the Optimizer (if you want to skew it later)
            bx, by, bw, bh = new_p['bbox']
            new_p['polygon'] = [
                [bx, by], [bx+bw, by], [bx+bw, by+bh], [bx, by+bh]
            ]
            
            final_panels.append(new_p)
            
        return final_panels