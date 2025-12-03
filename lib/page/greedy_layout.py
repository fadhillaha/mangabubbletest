import math

class PageLayoutEngine:
    def __init__(self, page_width=500, page_height=750, gutter=10):
        self.page_width = page_width
        self.page_height = page_height
        self.gutter = gutter

    def layout_page(self, panels):
        if not panels:
            return []

        # 1. Group panels into Rows based on Score
        rows = self._group_into_rows(panels)
        
        # 2. Calculate Height for each Row
        # Taller rows for "heavier" content
        total_page_score = sum(self._get_row_weight(r) for r in rows)
        current_y = 0
        final_panels = []

        for row_panels in rows:
            row_weight = self._get_row_weight(row_panels)
            
            # Row Height is proportional to its score
            # e.g., A row with score 12 gets more height than a row with score 8
            # Safety: Ensure row takes at least 15% of page
            ratio = max(0.15, row_weight / total_page_score) if total_page_score > 0 else 1.0
            row_height = self.page_height * ratio
            
            # 3. Layout the panels INSIDE this row (Side-by-Side)
            row_rect = [0, current_y, self.page_width, row_height]
            laid_out_row = self._layout_row_items(row_rect, row_panels)
            final_panels.extend(laid_out_row)
            
            current_y += row_height

        return final_panels

    def _get_row_weight(self, row):
        return sum(p.get('importance_score', 5) for p in row)

    def _group_into_rows(self, panels):
        """
        Groups a list of panels into sub-lists (rows) based on a target score.
        """
        total_score = sum(p.get('importance_score', 5) for p in panels)
        count = len(panels)
        
        # Heuristic: How many rows do we want?
        if count <= 2: num_rows = 1
        elif count <= 4: num_rows = 2
        elif count <= 8: num_rows = 3
        else: num_rows = 4

        target_score_per_row = total_score / num_rows
        
        rows = []
        current_row = []
        current_row_score = 0
        
        for p in panels:
            p_score = p.get('importance_score', 5)
            
            # Logic: If adding this panel exceeds the target significantly, 
            # and we already have items, start a new row.
            # We allow going slightly over (1.2x) to prevent stragglers.
            if current_row and (current_row_score + p_score > target_score_per_row * 1.2):
                rows.append(current_row)
                current_row = [p]
                current_row_score = p_score
            else:
                current_row.append(p)
                current_row_score += p_score
                
        if current_row:
            rows.append(current_row)
            
        return rows

    def _layout_row_items(self, row_rect, panels):
        """
        Splits a single Row Rectangle into columns for each panel.
        Adjusted for Japanese Manga Reading Order (Right-to-Left).
        """
        rx, ry, rw, rh = row_rect
        final_items = []
        
        row_total_score = self._get_row_weight(panels)
        
        # --- FIX: Start from the Right Edge ---
        current_x = rx + rw 
        
        for p in panels:
            p_score = p.get('importance_score', 5)
            
            # Width Ratio
            ratio = p_score / row_total_score if row_total_score > 0 else 1.0
            p_width = rw * ratio
            
            # --- FIX: Move Leftwards ---
            # 1. Shift pointer to the left edge of this new panel
            current_x -= p_width
            
            # 2. Use this as the starting X coordinate
            p_new = p.copy()
            p_new['bbox'] = [
                int(current_x + self.gutter),
                int(ry + self.gutter),
                int(p_width - (self.gutter * 2)),
                int(rh - (self.gutter * 2))
            ]
            final_items.append(p_new)
            
        return final_items