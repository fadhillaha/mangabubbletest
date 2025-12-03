# lib/page/bsp_layout.py

class BSPLayoutEngine:
    def __init__(self, page_width=1000, page_height=1414, gutter=10):
        self.page_width = page_width
        self.page_height = page_height
        self.gutter = gutter

    def layout_page(self, panels):
        if not panels: return []

        full_rect = [0, 0, self.page_width, self.page_height]
        
        # Start the recursive split
        layout_map = self._bsp_split(full_rect, panels)
        
        final_panels = []
        for p in panels:
            p_new = p.copy()
            idx = p['panel_index']
            if idx in layout_map:
                x, y, w, h = layout_map[idx]
                # Apply Gutter
                p_new['bbox'] = [
                    int(x + self.gutter),
                    int(y + self.gutter),
                    int(w - (self.gutter * 2)),
                    int(h - (self.gutter * 2))
                ]
            else:
                p_new['bbox'] = [0,0,10,10]
            final_panels.append(p_new)
            
        return final_panels

    def _get_total_weight(self, panels):
        return sum(p.get('importance_score', 5) for p in panels)

    def _decide_split_direction(self, rect, panels):
        """
        Decides whether to cut Horizontally (True) or Vertically (False).
        """
        _, _, w, h = rect
        aspect_ratio = w / h if h > 0 else 1.0

        # --- PRIORITY 1: Canvas Constraints (Fixes Skyscraper Effect) ---
        # If the box is noticeably taller than it is wide (Portrait), cut Horizontally.
        # This forces the page to break into rows first.
        if aspect_ratio < 0.8:  
            return True # Force Horizontal Cut

        # If the box is very wide (Landscape), cut Vertically.
        if aspect_ratio > 1.5:
            return False # Force Vertical Cut

        # --- PRIORITY 2: Metadata Voting ---
        # If the box is roughly square-ish, let the content decide.
        votes_tall = sum(1 for p in panels if p.get('suggested_aspect_ratio') == 'tall')
        votes_wide = sum(1 for p in panels if p.get('suggested_aspect_ratio') == 'wide')

        if votes_tall > votes_wide:
            return False # Vertical (allows panels to stand tall side-by-side)
        elif votes_wide > votes_tall:
            return True # Horizontal (stacks them)

        # Tie-breaker: Cut the longest side
        return h > w

    def _bsp_split(self, rect, panels):
        x, y, w, h = rect
        
        # Base Case
        if len(panels) == 1:
            return {panels[0]['panel_index']: rect}
        if len(panels) == 0:
            return {}

        # 1. Decide Split Direction
        split_horizontal = self._decide_split_direction(rect, panels)

        # 2. Find Split Index (Balanced Weight)
        total_weight = self._get_total_weight(panels)
        half_weight = total_weight / 2
        current_weight = 0
        split_index = 1
        closest_diff = float('inf')

        for i, p in enumerate(panels[:-1]):
            current_weight += p.get('importance_score', 5)
            diff = abs(current_weight - half_weight)
            if diff < closest_diff:
                closest_diff = diff
                split_index = i + 1
        
        # 3. Create Groups
        group_1 = panels[:split_index]
        group_2 = panels[split_index:]
        
        weight_1 = self._get_total_weight(group_1)
        weight_2 = self._get_total_weight(group_2)
        total_w = weight_1 + weight_2
        ratio = weight_1 / total_w if total_w > 0 else 0.5

        # 4. Calculate Coordinates & Recursion
        results = {}
        
        if split_horizontal:
            # Horizontal Split (Top / Bottom)
            # Top gets First Group (Standard Reading)
            h_1 = h * ratio
            h_2 = h - h_1
            rect_top = [x, y, w, h_1]
            rect_bottom = [x, y + h_1, w, h_2]
            
            results.update(self._bsp_split(rect_top, group_1))
            results.update(self._bsp_split(rect_bottom, group_2))
            
        else:
            # Vertical Split (Left / Right)
            w_1 = w * ratio
            w_2 = w - w_1
            
            # --- MANGA RTL LOGIC ---
            # Group 1 (Earlier panels) goes to the RIGHT
            # Group 2 (Later panels) goes to the LEFT
            rect_right = [x + w_2, y, w_1, h] # Right Side
            rect_left  = [x,       y, w_2, h] # Left Side
            
            results.update(self._bsp_split(rect_right, group_1))
            results.update(self._bsp_split(rect_left, group_2))

        return results