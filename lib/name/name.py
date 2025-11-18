from PIL import Image, ImageDraw, ImageFont
import os
from lib.layout.layout import MangaLayout, Speaker, NonSpeaker
from math import atan2, cos, sin, hypot
import random
from math import ceil

FONTPATH = "fonts/NotoSansCJK-Regular.ttc"


def draw_vertical_text(img, text, bbox, type):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONTPATH, 20)
    vertical_margin = 2
    horiziontal_margin = 2
    image_width, image_height = img.width, img.height
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    char_bbox = font.getbbox("あ")
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]
    chars_per_col = int(height / (char_height + vertical_margin))

    estimated_width = char_width * (int((len(text) / chars_per_col)) + 1)
    final_bbox = [bbox[2] - estimated_width, bbox[1], bbox[2], bbox[3]]

    def pad(bbox, padding_x, padding_y, image_width, image_height):
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(image_width, x2 + padding_x)
        y2 = min(image_height, y2 + padding_y)
        return [x1, y1, x2, y2]

    if type == "monologue":
        draw.rectangle(
            pad(final_bbox, 5, 5, image_width, image_height),
            fill="white",
            outline="black",
            width=2,
        )
    elif type == "dialogue":
        draw.ellipse(
            pad(final_bbox, 20, 30, image_width, image_height),
            fill="white",
            outline="black",
            width=2,
        )

    cur_col = 0
    cur_position = (bbox[2] - char_width, bbox[1])
    for char in text:
        draw.text(cur_position, char, font=font, fill="black")
        cur_col += 1
        if cur_col == chars_per_col:
            cur_col = 0
            cur_position = (cur_position[0] - char_width - horiziontal_margin, bbox[1])
        else:
            cur_position = (
                cur_position[0],
                cur_position[1] + char_height + vertical_margin,
            )

    return


def horizontal_paste(images):
    base_image = Image.new(
        "RGB",
        (sum(img.width for img in images), max(img.height for img in images)),
        (255, 255, 255),
    )
    for i, img in enumerate(images):
        base_image.paste(img, (sum(img.width for img in images[:i]), 0))
    return base_image


def _generate_name(pil_image, base_layout, scored_layout, panel):
    ref_layout, _, _ = scored_layout

    # QueryレイアウトのSpeakerの数 > 参照レイアウトのSpeakerの数 => 配置できないのでネームの生成は断念
    num_speakers_in_ref_layout = 0
    for layout_ele in ref_layout.elements:
        if type(layout_ele) == Speaker:
            num_speakers_in_ref_layout += 1
    num_speakers = 0
    for layout_ele in base_layout.elements:
        if type(layout_ele) == Speaker:
            num_speakers += 1
    if num_speakers_in_ref_layout < num_speakers:
        print("num_speakers_in_ref_layout < num_speakers")
        return False

    # 参照レイアウトのSpeakerエレメントを吹き出しの登場順に並び替え
    def custom_sort(element):
        if type(element) == NonSpeaker:
            return -1
        if element.text_info is None:
            return 0
        center_pos = []
        for text_obj in element.text_info:
            center_pos.append((text_obj["bbox"][0] + text_obj["bbox"][2]) / 2)
        return max(center_pos)

    ref_layout.elements = sorted(ref_layout.elements, key=custom_sort, reverse=True)

    # セリフの位置を参照レイアウトのエレメントをもとに割り当て
    # 1. QueryレイアウトのSpeakerの数 <= 参照レイアウトのSpeakerの数
    # 2. 前述の並び替えでSpeakerエレメントはNonSpeakerよりも必ず前側に来る
    # => NonSpeakerエレメントがセリフ位置の参照に使われることはない
    pairs = scored_layout[2] 

    # 2. Map your script dialogue to your Base Speakers (0, 1, 2...)
    # (This is safe because generate_layout created them in this exact order)
    dialogue_map = {}
    speaker_idx = 0
    for element in panel:
        if element["type"] == "dialogue":
            dialogue_map[speaker_idx] = element["content"]
            speaker_idx += 1

    # 3. Loop through the PAIRS to assign text
    # base_idx = Your Speaker Index (0=Boy, 1=Girl)
    # ref_idx  = Template Speaker Index (Correct matching spatial position)
    for base_idx, ref_idx in pairs:
        
        # Get the dialogue for YOUR speaker
        if base_idx not in dialogue_map: continue
        text = dialogue_map[base_idx]

        # Get the corresponding TEMPLATE speaker
        template_speaker = ref_layout.elements[ref_idx]
        
        # Safety check: ensure the template element is actually a speaker with text info
        if type(template_speaker) != Speaker or not template_speaker.text_info:
            continue
        
        # Find that template speaker's bubble (using the simple rightmost logic for now)
        bbox = [-1, -1, -1, -1]
        for text_obj in template_speaker.text_info:
            if text_obj["bbox"][2] > bbox[2]:
                bbox = text_obj["bbox"]
        
        # Draw the text in the CORRECT bubble
        # Note: If you implemented the "tail" improvement, pass character_bbox here!
        # character_bbox = base_layout.elements[base_idx].bbox
        # draw_vertical_text(pil_image, text, bbox, "dialogue", character_bbox)
        
        draw_vertical_text(pil_image, text, bbox, "dialogue")

    # monologueはpanel上は複数に分かれていても１つとして扱う
    total_monologue = ""
    for panel_ele in panel:
        if panel_ele["type"] == "monologue":
            total_monologue += panel_ele["content"]

    unrelated_text_bboxes = [bbox["bbox"] for bbox in ref_layout.unrelated_text_bbox]
    unrelated_text_bboxes = sorted(
        unrelated_text_bboxes, key=lambda x: x[2], reverse=True
    )
    if len(ref_layout.unrelated_text_bbox) > 0:
        draw_vertical_text(
            pil_image, total_monologue, unrelated_text_bboxes[0], "monologue"
        )
    return


def generate_name(controlnet_result, base_layout, scored_layout, panel, save_path):
    width, height = controlnet_result.canvas_width, controlnet_result.canvas_height
    # pose_only_pil_image = controlnetres2pil(controlnet_result)
    original_pil_image = Image.open(controlnet_result.base_image_path).resize(
        (width, height)
    )
    # _generate_name(pose_only_pil_image, base_layout, scored_layout, panel)
    # _generate_name(controlnet_res_pil_image, base_layout, scored_layout, panel)
    _generate_name(original_pil_image, base_layout, scored_layout, panel)
    manga_layout_image = Image.open(scored_layout[0].image_path).resize((width, height))
    images = [manga_layout_image, original_pil_image]
    original_pil_image.save(save_path.replace(".png", "_onlyname.png"))
    horizontal_pasted_image = horizontal_paste(images)
    horizontal_pasted_image.save(save_path)


def generate_animepose_image(base_image_path, prompt, save_path):
    from lib.image.controlnet import generate_with_controlnet_openpose

    prefix = "(<lora:animeoutlineV4_16:1>), monochrome, monochrome, storyboard_style, clean_lineart"
    generate_with_controlnet_openpose(base_image_path, prefix + prompt, save_path)
