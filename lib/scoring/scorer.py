import os
import torch
import json
from PIL import Image, ImageFont
from transformers import CLIPProcessor, CLIPModel
from lib.layout.layout import Speaker
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Font used for speech bubble
FONTPATH = "fonts/NotoSansCJK-Regular.ttc"

# ---   CLIP MODEL & PROMPT LOGIC   ---

def load_clip_model(model_id="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Scoring] Loading CLIP model: {model_id} on {device}...")
    try:
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor, device
    except Exception as e:
        print(f"[Scoring] Failed to load CLIP model: {e}")
        return None, None, None

def get_verification_prompt(text_prompt):
    """
    Appends stylistic keywords added in lib/name.py
    Args:
        text_prompt (str): The full, enhanced prompt used for generation.
    """
    style_suffix = ", rough pencil sketch, manga name, storyboard style, loose lines, messy drawing, monochrome"
    return text_prompt + style_suffix

def calculate_clip_score(image_path, text_prompt, model, processor, device):
    if model is None or not os.path.exists(image_path): return 0.0
    try:
        image = Image.open(image_path).convert("RGB")
        
        # 1. Tokenize without truncation first (to handle long prompts manually)
        inputs_text = processor(text=[text_prompt], return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs_text['input_ids'][0]

        # 2. Split into chunks of 75 tokens (leaving room for start/end tokens)
        chunk_size = 75
        chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
        
        text_embeds_list = []
        
        # 3. Process each chunk
        with torch.no_grad():
            # Get image embedding
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_outputs = model.get_image_features(**image_inputs)
            image_embeds = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)

            for chunk in chunks:

                bos_id = processor.tokenizer.bos_token_id
                eos_id = processor.tokenizer.eos_token_id
                
                chunk_tensor = torch.cat([
                    torch.tensor([bos_id]), 
                    chunk, 
                    torch.tensor([eos_id])
                ]).unsqueeze(0).to(device)

                # Get text features
                text_outputs = model.get_text_features(input_ids=chunk_tensor)
                text_embeds = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
                text_embeds_list.append(text_embeds)

        # 4. Average the text embeddings
        if text_embeds_list:
            avg_text_embed = torch.mean(torch.stack(text_embeds_list), dim=0)
            # Normalize again after averaging
            avg_text_embed = avg_text_embed / avg_text_embed.norm(p=2, dim=-1, keepdim=True)
            
            # Calculate final score
            return torch.matmul(avg_text_embed, image_embeds.t()).item()
        else:
            return 0.0

    except Exception as e:
        print(f"[Scoring] Error calculating CLIP score: {e}")
        return 0.0

# ---   GEOMETRIC PENALTY LOGIC   ---

def _get_font_metrics():
    """Returns font object and estimated char size for bubble simulation."""
    try:
        font = ImageFont.truetype(FONTPATH, 20)
        char_bbox = font.getbbox("あ")
        return font, char_bbox[2] - char_bbox[0], char_bbox[3] - char_bbox[1]
    except:
        return None, 20, 20

def _simulate_dialogue_bboxes(ref_layout, dialogues):
    """Simulate bubbles placement for characters (Speakers)."""
    bubbles = []
    _, char_width, char_height = _get_font_metrics()
    vertical_margin = 2

    ref_speakers = [ele for ele in ref_layout.elements if isinstance(ele, Speaker)]
    
    def custom_sort(element):
        if element.text_info is None: return 0
        center_pos = []
        for text_obj in element.text_info:
            center_pos.append((text_obj["bbox"][0] + text_obj["bbox"][2]) / 2)
        return max(center_pos) if center_pos else 0

    ref_speakers = sorted(ref_speakers, key=custom_sort, reverse=True)
    
    panel_pointer = 0
    for ref_element in ref_speakers:
        if panel_pointer >= len(dialogues): break
        
        ref_bbox = [-1, -1, -1, -1]
        if ref_element.text_info:
            for text_obj in ref_element.text_info:
                if text_obj["bbox"][2] > ref_bbox[2]:
                    ref_bbox = text_obj["bbox"]
        
        if ref_bbox[0] == -1:
            panel_pointer += 1
            continue

        text_content = dialogues[panel_pointer]
        box_height = ref_bbox[3] - ref_bbox[1]
        chars_per_col = int(box_height / (char_height + vertical_margin))
        if chars_per_col < 1: chars_per_col = 1
        
        num_cols = int(len(text_content) / chars_per_col) + 1
        estimated_width = char_width * num_cols
        
        bubbles.append([ref_bbox[2] - estimated_width, ref_bbox[1], ref_bbox[2], ref_bbox[3]])
        panel_pointer += 1
        
    return bubbles

def _simulate_monologue_bbox(ref_layout, monologue_text):
    """Simulates the bubble placement for the Monologue (Unrelated Text)."""
    if not monologue_text or not hasattr(ref_layout, 'unrelated_text_bbox') or not ref_layout.unrelated_text_bbox:
        return None

    _, char_width, char_height = _get_font_metrics()
    vertical_margin = 2

    # Find the best slot for monologue
    unrelated_bboxes = sorted([item["bbox"] for item in ref_layout.unrelated_text_bbox], key=lambda x: x[2], reverse=True)
    
    if not unrelated_bboxes:
        return None
        
    target_bbox = unrelated_bboxes[0]
    
    box_height = target_bbox[3] - target_bbox[1]
    chars_per_col = int(box_height / (char_height + vertical_margin))
    if chars_per_col < 1: chars_per_col = 1
    
    num_cols = int(len(monologue_text) / chars_per_col) + 1
    estimated_width = char_width * num_cols
    
    # Monologues are also anchored right
    return [target_bbox[2] - estimated_width, target_bbox[1], target_bbox[2], target_bbox[3]]

def _calculate_intersection(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def _get_face_bbox(person, width, height):
    if not person.face_keypoints_2d: return None
    xs = [kp[0] for kp in person.face_keypoints_2d if kp[0] > 0]
    ys = [kp[1] for kp in person.face_keypoints_2d if kp[1] > 0]
    if not xs or not ys: return None
    return [min(xs)*width, min(ys)*height, max(xs)*width, max(ys)*height]

def _visualize_penalty_debug(people_result, bubbles, save_path, penalty=0):
    """
    Generates a visual map of the geometric penalty check.
    Green = Body Box, Blue Dots = Joints, Orange = Face, Red = Text Bubble.
    """
    if not save_path: return

    try:
        width = people_result.canvas_width
        height = people_result.canvas_height
        
        fig, ax = plt.subplots(figsize=(6, 6 * height / width), dpi=100)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0) 
        ax.set_aspect('equal')
        
        # 1. Draw People
        for person in people_result.people:
            # A. Draw Joints (Blue Dots) 
            if person.pose_keypoints_2d:
                for kp in person.pose_keypoints_2d:
                    # Exception if OpenPose returns 0,0 for undetected points
                    if kp[0] > 0 and kp[1] > 0:
                        ax.plot(kp[0] * width, kp[1] * height, 'o', color='blue', markersize=4, alpha=0.8)

            # B. Draw Body Box (Green)
            if person.pose_keypoints_2d:
                pxs = [kp[0] for kp in person.pose_keypoints_2d if kp[0] > 0]
                pys = [kp[1] for kp in person.pose_keypoints_2d if kp[1] > 0]
                if pxs and pys:
                    x1, y1 = min(pxs)*width, min(pys)*height
                    x2, y2 = max(pxs)*width, max(pys)*height
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                             linewidth=2, edgecolor='green', facecolor='none', 
                                             linestyle=':', label='Body Box')
                    ax.add_patch(rect)
            
            # C. Draw Face Box (Orange)
            f_bbox = _get_face_bbox(person, width, height)
            if f_bbox:
                rect = patches.Rectangle((f_bbox[0], f_bbox[1]), f_bbox[2]-f_bbox[0], f_bbox[3]-f_bbox[1], 
                                         linewidth=2, edgecolor='orange', facecolor='none', label='Face')
                ax.add_patch(rect)
        
        # 2. Draw Bubbles (Red Dashed)
        for i, b in enumerate(bubbles):
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], 
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Bubble')
            ax.add_patch(rect)
            ax.text(b[0], b[1]-5, "Text", color='red', fontsize=8, weight='bold')

        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Add a fake handle for the dots to show in legend
        if 'Body Box' in by_label: 
            import matplotlib.lines as mlines
            dot_handle = mlines.Line2D([], [], color='blue', marker='o', markersize=4, linestyle='None', label='Joints')
            by_label['Joints'] = dot_handle

        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

        plt.title(f"Penalty = {penalty}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"[Scoring] Failed to visualize geometric penalty: {e}")

def calculate_geometric_penalty(ref_layout, panel_data, people_result, bbox_save_path=None):
    """
    Calculates penalty for both Dialogues and Monologues overlapping faces/bodies.
    Args:
        ref_layout: The MangaLayout template.
        panel_data: The list of dictionaries (the raw panel content).
        people_result: OpenPose result.
    """
    if not ref_layout or not people_result or not people_result.people:
        return 0.0

    # 1. Extract Texts
    dialogues = [ele["content"] for ele in panel_data if ele["type"] == "dialogue"]
    monologues = [ele["content"] for ele in panel_data if ele["type"] == "monologue"]
    total_monologue = "".join(monologues)

    # 2. Simulate All Bubbles
    bubbles = _simulate_dialogue_bboxes(ref_layout, dialogues)
    mono_bbox = _simulate_monologue_bbox(ref_layout, total_monologue)
    if mono_bbox:
        bubbles.append(mono_bbox) # Add monologue to the check list

    width = people_result.canvas_width
    height = people_result.canvas_height
    total_penalty = 0.0

    for b_bbox in bubbles:
        b_area = (b_bbox[2] - b_bbox[0]) * (b_bbox[3] - b_bbox[1])
        if b_area <= 0: continue

        for person in people_result.people:
            # Body Overlap (Weight: 1.0)
            if person.pose_keypoints_2d:
                pxs = [kp[0] for kp in person.pose_keypoints_2d if kp[0] > 0]
                pys = [kp[1] for kp in person.pose_keypoints_2d if kp[1] > 0]
                if pxs and pys:
                    p_bbox = [min(pxs)*width, min(pys)*height, max(pxs)*width, max(pys)*height]
                    intersect = _calculate_intersection(b_bbox, p_bbox)
                    if intersect > 0:
                        total_penalty += (intersect / b_area) * 100 * 1.0

            # Face Overlap (Weight: 5.0)
            f_bbox = _get_face_bbox(person, width, height)
            if f_bbox:
                intersect = _calculate_intersection(b_bbox, f_bbox)
                if intersect > 0:
                    total_penalty += (intersect / b_area) * 100 * 5.0

    if bbox_save_path:
        _visualize_penalty_debug(people_result, bubbles, bbox_save_path, total_penalty)

    return total_penalty

def run_panel_scoring(base_dir, prompts):
    """
    Runs the final scoring pass on all generated panels.
    Calculates CLIP scores and combines them with geometric penalties.
    Updates the scores.json files in place.
    """
    image_base_dir = os.path.join(base_dir, "images")
    
    # 1. Load Model (Heavy operation, done once)
    clip_model, clip_processor, device = load_clip_model()
    if not clip_model:
        print("Skipping scoring due to model load failure.")
        return

    # 2. Iterate through panels
    for i, prompt in enumerate(prompts):
        panel_dir = os.path.join(image_base_dir, f"panel{i:03d}")
        score_file = os.path.join(panel_dir, "scores.json")
        
        if not os.path.exists(score_file):
            continue

        try:
            with open(score_file, "r", encoding="utf-8") as f:
                panel_entry = json.load(f)
        except Exception as e:
            print(f"Error loading scores for panel {i}: {e}")
            continue

        # Prepare Verification Prompt
        ver_prompt = get_verification_prompt(prompt)
        
        best_score = -9999
        winner_name = "None"

        print(f"Scoring Panel {i}...")

        for var in panel_entry["variations"]:
            # A. Calculate CLIP Score
            c_score = calculate_clip_score(
                var["anime_image_path"], 
                ver_prompt, 
                clip_model, 
                clip_processor, 
                device
            )
            var["clip_score"] = c_score

            # B. Calculate Final Score
            for layout_opt in var["layout_options"]:
                sim = layout_opt["sim_score"]
                geom = layout_opt["geom_penalty"]
                
                # Layout Similarity + CLIP - Geometric Penalty
                final_score = (sim * 10) + (c_score * 50) - (geom)
                layout_opt["final_score"] = final_score
                
                # Track winner
                if final_score > best_score:
                    best_score = final_score
                    fname = os.path.basename(layout_opt.get("generated_image_path", "??"))
                    winner_name = f"Var {var['variation_id']} / {fname}"

        # 3. Save updates
        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(panel_entry, f, indent=4, default=str)
            
        print(f"Winner: {winner_name} (Score: {best_score:.2f})")