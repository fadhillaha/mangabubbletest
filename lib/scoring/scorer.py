import os
import torch
from PIL import Image, ImageFont
from transformers import CLIPProcessor, CLIPModel
from lib.layout.layout import Speaker

FONTPATH = "fonts/NotoSansCJK-Regular.ttc"

# ==========================================
# 1. CLIP MODEL MANAGEMENT
# ==========================================

def load_clip_model(model_id="openai/clip-vit-base-patch32"):
    """Loads the CLIP model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Scoring] Loading CLIP model: {model_id} on {device}...")

    try:
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        print("[Scoring] CLIP model loaded successfully.")
        return model, processor, device
    except Exception as e:
        print(f"[Scoring] Failed to load CLIP model: {e}")
        return None, None, None

def get_verification_prompt(panel_data):
    """Constructs the text prompt for CLIP with style tags."""
    descriptions = [ele['content'] for ele in panel_data if ele['type'] == 'description']
    scene_text = " ".join(descriptions)
    
    if not scene_text:
        dialogues = [ele['content'] for ele in panel_data if ele['type'] == 'dialogue']
        scene_text = " ".join(dialogues)

    if len(scene_text) > 200:
        scene_text = scene_text[:200] + "..."

    style_suffix = ", rough pencil sketch, manga name, storyboard style, loose lines, messy drawing, monochrome"
    return scene_text + style_suffix

def calculate_clip_score(image_path, text_prompt, model, processor, device):
    """Calculates Cosine Similarity between image and text."""
    if model is None or not os.path.exists(image_path):
        return 0.0

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text_prompt], 
            images=image, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        return torch.matmul(text_embeds, image_embeds.t()).item()
        
    except Exception as e:
        print(f"[Scoring] Error calculating CLIP score: {e}")
        return 0.0

# ==========================================
# 2. GEOMETRIC PENALTY LOGIC
# ==========================================

def _simulate_bubble_bboxes(ref_layout, dialogues):
    """Internal helper to calculate where bubbles would go."""
    bubbles = []
    try:
        font = ImageFont.truetype(FONTPATH, 20)
        char_bbox = font.getbbox("あ")
        char_width = char_bbox[2] - char_bbox[0]
        char_height = char_bbox[3] - char_bbox[1]
        vertical_margin = 2
    except Exception:
        char_width = 20
        char_height = 20
        vertical_margin = 2
        font = None

    # Sort speakers to match reading order/pipeline logic
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
        
        # Find largest text slot
        ref_bbox = [-1, -1, -1, -1]
        if ref_element.text_info:
            for text_obj in ref_element.text_info:
                if text_obj["bbox"][2] > ref_bbox[2]:
                    ref_bbox = text_obj["bbox"]
        
        if ref_bbox[0] == -1:
            panel_pointer += 1
            continue

        # Calculate size
        text_content = dialogues[panel_pointer]
        box_height = ref_bbox[3] - ref_bbox[1]
        chars_per_col = int(box_height / (char_height + vertical_margin))
        if chars_per_col < 1: chars_per_col = 1
        
        num_cols = int(len(text_content) / chars_per_col) + 1
        estimated_width = char_width * num_cols
        
        final_x2 = ref_bbox[2]
        final_x1 = ref_bbox[2] - estimated_width
        final_y1 = ref_bbox[1]
        final_y2 = ref_bbox[3]
        
        bubbles.append([final_x1, final_y1, final_x2, final_y2])
        panel_pointer += 1
        
    return bubbles

def _calculate_intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    return (x_right - x_left) * (y_bottom - y_top)

def _get_face_bbox(person, width, height):
    if not person.face_keypoints_2d: return None
    xs = [kp[0] for kp in person.face_keypoints_2d]
    ys = [kp[1] for kp in person.face_keypoints_2d]
    return [min(xs)*width, min(ys)*height, max(xs)*width, max(ys)*height]

def calculate_geometric_penalty(ref_layout, dialogues, people_result):
    """
    Calculates the total penalty score for a layout based on overlaps.
    Automatically simulates bubble positions based on the text.
    """
    if not ref_layout or not people_result or not people_result.people:
        return 0.0

    # 1. Simulate where the bubbles will be
    bubbles = _simulate_bubble_bboxes(ref_layout, dialogues)
    
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
                p_bbox = [min(pxs)*width, min(pys)*height, max(pxs)*width, max(pys)*height]
                
                intersect = _calculate_intersection_area(b_bbox, p_bbox)
                if intersect > 0:
                    total_penalty += (intersect / b_area) * 100 * 1.0

            # Face Overlap (Weight: 5.0)
            f_bbox = _get_face_bbox(person, width, height)
            if f_bbox:
                intersect = _calculate_intersection_area(b_bbox, f_bbox)
                if intersect > 0:
                    total_penalty += (intersect / b_area) * 100 * 5.0

    return total_penalty