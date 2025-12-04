import scipy.io
import numpy as np
import json
import os
import glob
import sys
import h5py

class StyleTrainer:
    def __init__(self):
        # 1. Structure Model: Splits (H vs V)
        self.structure_counts = {}
        
        # 2. Importance Model: Target Area % for each Rank
        self.importance_data = {} 

        # 3. Shape Model (NEW): Vertex Deltas for non-rectangular shapes
        # We store (dx, dy) for all 4 corners relative to a normalized 1x1 box
        self.vertex_deltas = [] 

    def train_from_folder(self, folder_path):
        search_path = os.path.join(folder_path, "**", "*.mat")
        mat_files = glob.glob(search_path, recursive=True)
        print(f"Found {len(mat_files)} training pages.")

        for fpath in mat_files:
            try:
                self._process_file_robust(fpath)
            except Exception as e:
                print(f"Skipping {os.path.basename(fpath)}: {e}")

        output_path = os.path.join(os.path.dirname(__file__), "style_models.json")
        self._save_models(output_path)

    def _process_file_robust(self, fpath):
        # Try Scipy first, then H5PY
        try:
            data = scipy.io.loadmat(fpath)
            if 'page' in data:
                self._extract_from_struct(data['page'])
        except (NotImplementedError, ValueError):
            with h5py.File(fpath, 'r') as f:
                if 'page' in f:
                    self._extract_from_h5(f['page'])

    def _extract_from_struct(self, page_struct):
        try:
            # Handle varied nesting of .mat files
            if 'P' in page_struct.dtype.names:
                raw_panels = page_struct['P'][0, 0]
            else:
                return
            
            panels = []
            for i in range(raw_panels.size):
                poly = raw_panels[i]
                if poly.size > 0:
                    panels.append(poly) # Keep raw polygon for Shape learning
            
            self._process_panels(panels)
        except: pass

    def _extract_from_h5(self, page_grp):
        try:
            if 'P' not in page_grp: return
            p_refs = page_grp['P']
            panels = []
            for ref_row in p_refs:
                for ref in ref_row:
                    try:
                        ds = page_grp.file[ref]
                        data = ds[:]
                        if data.shape[0] == 2 and data.shape[1] > 2: data = data.T
                        panels.append(data)
                    except: continue
            self._process_panels(panels)
        except: pass

    def _process_panels(self, raw_polygons):
        if not raw_polygons: return

        # 1. Normalize Page to 0..1
        all_pts = np.vstack(raw_polygons)
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)
        page_w, page_h = max_x - min_x, max_y - min_y
        
        if page_w == 0 or page_h == 0: return

        normalized_panels = []
        
        for poly in raw_polygons:
            # Normalize coordinates
            norm_poly = np.copy(poly)
            norm_poly[:,0] = (poly[:,0] - min_x) / page_w
            norm_poly[:,1] = (poly[:,1] - min_y) / page_h
            
            # AABB for Structure/Importance
            p_min_x, p_min_y = np.min(norm_poly, axis=0)
            p_max_x, p_max_y = np.max(norm_poly, axis=0)
            w, h = p_max_x - p_min_x, p_max_y - p_min_y
            area = w * h
            
            normalized_panels.append({
                "bbox": {"x": p_min_x, "y": p_min_y, "w": w, "h": h},
                "area": area,
                "poly": norm_poly # Store normalized polygon
            })

            # --- COLLECT SHAPE DATA (Vertex Deltas) ---
            # Measure deviation from a perfect rectangle
            # We assume 4 corners roughly map to TL, TR, BR, BL
            if len(norm_poly) == 4:
                # Sort corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
                # Simple sort by Y then X usually works for convex manga panels
                center = np.mean(norm_poly, axis=0)
                angles = np.arctan2(norm_poly[:,1] - center[1], norm_poly[:,0] - center[0])
                sorted_indices = np.argsort(angles)
                sorted_poly = norm_poly[sorted_indices]
                
                # Re-order to TL, TR, BR, BL (Angles: -135, -45, 45, 135 roughly)
                # This part is heuristic; for robustness we map to closest ideal corner
                ideals = np.array([
                    [p_min_x, p_min_y], [p_max_x, p_min_y], 
                    [p_max_x, p_max_y], [p_min_x, p_max_y]
                ])
                
                deltas = []
                for i in range(4):
                    # Find closest actual point to this ideal corner
                    dists = np.linalg.norm(sorted_poly - ideals[i], axis=1)
                    closest_idx = np.argmin(dists)
                    pt = sorted_poly[closest_idx]
                    
                    # Store delta relative to Width/Height (Resolution Independent)
                    dx = (pt[0] - ideals[i][0]) / (w if w>0 else 1)
                    dy = (pt[1] - ideals[i][1]) / (h if h>0 else 1)
                    deltas.extend([dx, dy])
                
                self.vertex_deltas.append(deltas)

        # --- COLLECT IMPORTANCE & STRUCTURE ---
        # Sort by Area
        sorted_panels = sorted(normalized_panels, key=lambda x: x['area'], reverse=True)
        num_p = len(sorted_panels)
        
        if num_p not in self.importance_data: self.importance_data[num_p] = {}
        for r, p in enumerate(sorted_panels):
            if r not in self.importance_data[num_p]: self.importance_data[num_p][r] = []
            self.importance_data[num_p][r].append(p['area'])

        # Tree Recovery (Simplified XY-Cut for Structure Learning)
        # We only use the bounding boxes for this part
        bboxes = [p['bbox'] for p in normalized_panels]
        self._recover_tree(bboxes, depth=0)

    def _recover_tree(self, panels, depth):
        if len(panels) <= 1: return

        # Sort by Y and X
        by_y = sorted(panels, key=lambda p: p['y'])
        by_x = sorted(panels, key=lambda p: p['x'])
        
        # Determine Split
        split_h = self._find_gap(by_y, 'y', 'h')
        split_v = self._find_gap(by_x, 'x', 'w')
        
        split_type = None
        if split_h != -1: split_type = "H"
        elif split_v != -1: split_type = "V"
        
        if split_type:
            key = f"depth_{depth}"
            if key not in self.structure_counts: self.structure_counts[key] = {"H":0, "V":0}
            self.structure_counts[key][split_type] += 1
            
            # Recurse
            if split_type == "H":
                self._recover_tree(by_y[:split_h+1], depth+1)
                self._recover_tree(by_y[split_h+1:], depth+1)
            else:
                self._recover_tree(by_x[:split_v+1], depth+1)
                self._recover_tree(by_x[split_v+1:], depth+1)

    def _find_gap(self, sorted_panels, coord, dim):
        for i in range(len(sorted_panels)-1):
            curr_end = sorted_panels[i][coord] + sorted_panels[i][dim]
            next_start = sorted_panels[i+1][coord]
            if next_start >= curr_end - 0.01: return i
        return -1

    def _save_models(self, filename):
        # 1. Structure
        final_struct = {}
        for d, c in self.structure_counts.items():
            tot = c["H"] + c["V"]
            final_struct[d] = {"H": c["H"]/tot, "V": c["V"]/tot} if tot>0 else {"H":0.5, "V":0.5}

        # 2. Importance
        final_imp = {}
        for n, ranks in self.importance_data.items():
            final_imp[n] = {r: sum(a)/len(a) for r, a in ranks.items()}

        # 3. Shape (Mean and Std of Vertex Deltas)
        # We stored 8 values per panel: [dx1, dy1, dx2, dy2, ...]
        deltas_arr = np.array(self.vertex_deltas)
        if len(deltas_arr) > 0:
            final_shape = {
                "mean": np.mean(deltas_arr, axis=0).tolist(),
                "std": np.std(deltas_arr, axis=0).tolist()
            }
        else:
            final_shape = {"mean": [0]*8, "std": [0.05]*8}

        with open(filename, "w") as f:
            json.dump({
                "structure": final_struct,
                "importance": final_imp,
                "shape": final_shape
            }, f, indent=4)
        print(f"Saved style models to {filename}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # option 1: Absolute Path (Easiest)
    dataset_path = "C:\\DATA\\Stuff\\Parsola\\Bubble\\dataset\\Data\\detective_conan"
    
    # option 2: Relative Path (Cleaner)
    # This assumes your 'dataset' folder is next to your 'layoutpreparation' folder
    # Get the folder where THIS script lives
    # current_dir = os.path.dirname(__file__)
    # # Go up one level to root, then into 'dataset/mat_files'
    # dataset_path = os.path.join(current_dir, "..", "dataset", "mat_files")
    
    # # Normalize the path (fixes slash directions on Windows/Mac)
    # dataset_path = os.path.abspath(dataset_path)

    print(f"Looking for training data in: {dataset_path}")

    # --- EXECUTION ---
    trainer = StyleTrainer()
    trainer.train_from_folder(dataset_path)