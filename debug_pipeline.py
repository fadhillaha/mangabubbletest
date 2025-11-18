import streamlit as st
import os
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from dotenv import load_dotenv

# --- 0. Setup & Imports ---
try:
    # Import your Adapter Client
    from lib.llm.geminiadapter import GeminiClient
    
    # Import the original library functions
    from lib.script.divide import divide_script, ele2panels, refine_elements
    from lib.image.image import generate_image_prompts, enhance_prompts, generate_image_with_sd
    from lib.image.controlnet import run_controlnet_openpose, controlnet2bboxes, ControlNetResult
    from lib.name.name import generate_animepose_image, generate_name
    from lib.layout.layout import generate_layout, similar_layouts, Speaker, NonSpeaker
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.error("Make sure you run this script from the project root and have run 'pip install -e .'")
    st.stop()

# Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set in .env file")
    st.stop()

try:
    client = GeminiClient(api_key=api_key, model="gemini-2.0-flash")
except Exception as e:
    st.error(f"Failed to initialize GeminiClient: {e}")
    st.stop()

# Config
OUTPUT_DIR = "output_debug_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_dir = OUTPUT_DIR

st.set_page_config(page_title="Full Pipeline Debugger", layout="wide")
st.title("🚀 Full Pipeline Debugger (Gemini Adapter)")

# --- 1. Script Input ---
st.sidebar.header("Input Script")
default_script = """放課後の教室。夕焼けが窓から差し込み、室内をオレンジ色に染めている。 誰もいない教室の隅で、ケンジが机に突っ伏して頭を抱えている。



ケンジ「もうダメだ…終わった…。」



ケンジの席の隣に立ち、呆れたように見下ろしているアイコ。



アイコ「大袈裟ね。たかが数学の宿題でしょ？」



（アイコの容姿） 黒髪のロングヘア。整った制服姿。知的で少し勝ち気な目元。腕を組んでいる。



ケンジ、涙目で顔を上げ、アイコにすがるように手を伸ばす。



（ケンジの容姿） 茶髪でボサボサ頭。ネクタイが緩んでいる。情けない表情。



ケンジ「アイコ様、お願いします！ノート見せてください！」



アイコ「お断りよ。自分で解かないと意味ないでしょ。」



冷たく突き放され、机に突っ伏して泣き真似をするケンジ。



ケンジM「（ケチ！アイコの鬼！悪魔！…でも頼れるのはこいつしかいないんだよなぁ。）」



ガララッ！と勢いよく教室のドアが開く。 部活帰りのヒロが、汗を拭きながら入ってくる。



（ヒロの容姿） 短髪でスポーティー。シャツの袖をまくっている。首にタオル。元気な笑顔。



ヒロ「おっ、まだ残ってたのか！奇遇だな！」



ヒロ、ドカドカと歩み寄り、ケンジとアイコの間に割り込むようにして机に腰掛ける。 （画面構成：左にケンジ、中央にヒロ、右にアイコ）



アイコ「ヒロ、机に座らないでよ。行儀悪い。」



ヒロ「固いこと言うなよ学級委員長。それより見てくれよコレ！」



ヒロ、ポケットから少し潰れた「限定焼きそばパン」を得意げに取り出す。



ヒロ「購買のラス１、ゲットしたぜ！凄いだろ！」



パンを見た瞬間、アイコの表情が一変する。身を乗り出してパンを凝視する。



アイコ「嘘…それ、幻のプレミアム焼きそばパン！？」



ケンジ「（えっ、アイコってそんなキャラだっけ？）」



ヒロ、ニカっと笑ってパンを高く掲げる。 釣られて手を伸ばすアイコ。



ヒロ「へへーん、羨ましいだろ！一口もやらねーけどな！」



アイコ「ちょっと！一口くらい味見させてよ！ケチ！」



ヒロ「バーカ！早い者勝ちなんだよ！」



子供のようにパンを巡って争い始めるヒロとアイコ。 二人の間に挟まれ、置いてきぼりにされたケンジ。



ケンジ「あの…俺の宿題は…？」



ヒロとアイコ、同時にケンジの方を向く。



ヒロ＆アイコ「「うるさい！」」



ケンジ、再び机に突っ伏してふて寝する。



ケンジM「（…帰りたい。）」"
"""
script_text = st.sidebar.text_area("Enter Script", default_script, height=300)

if st.button("Run Full Pipeline", type="primary"):
    
    script_path = os.path.join(OUTPUT_DIR, "temp_script.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    st.header("1. Script Processing (LLM)")
    
    # --- Step 1: Script Parsing ---
    with st.status("Parsing Script...", expanded=True) as status:
        st.write("running `divide_script`...")
        elements = divide_script(client, script_path, base_dir)
        st.write("running `refine_elements`...")
        elements = refine_elements(elements, base_dir)
        
        speakers = list(set([e["speaker"] for e in elements if "speaker" in e]) - {""})
        st.success(f"Detected Speakers: {speakers}")

        st.write("running `ele2panels`...")
        panels = ele2panels(client, elements, base_dir)
        
        st.write("running `generate_image_prompts`...")
        prompts = generate_image_prompts(client, panels, speakers, base_dir)
        
        st.write("running `enhance_prompts`...")
        enhanced_prompts = enhance_prompts(client, prompts, base_dir)
        
        status.update(label="Script Processing Complete!", state="complete", expanded=False)

    st.header("2. Image Generation & Layout (All Panels)")
    
    progress_bar = st.progress(0)
    
    for panel_idx, panel_data in enumerate(panels):
        progress_bar.progress((panel_idx + 1) / len(panels))
        
        with st.expander(f"🖼️ Panel {panel_idx + 1} Processing", expanded=(panel_idx==0)):
            
            # 1. Get Prompt
            try:
                prompt_data = enhanced_prompts[panel_idx]
                final_prompt = prompt_data["prompt"] if isinstance(prompt_data, dict) else prompt_data
                st.info(f"**Prompt:** {final_prompt}")
            except IndexError:
                st.error(f"Error: No prompt found for Panel {panel_idx}")
                continue

            col1, col2 = st.columns(2)
            
            # --- Image Gen ---
            with col1:
                st.subheader("A. Generation")
                img_name = f"panel{panel_idx}_00.png"
                img_path = os.path.join(OUTPUT_DIR, img_name)
                anime_path = os.path.join(OUTPUT_DIR, f"panel{panel_idx}_00_anime.png")
                
                generate_image_with_sd(final_prompt, img_path)
                st.image(img_path, caption="Main Image")
                
                generate_animepose_image(img_path, final_prompt, anime_path)
                st.image(anime_path, caption="Sketch (for OpenPose)")

            # --- ControlNet ---
            with col2:
                st.subheader("B. Analysis")
                openpose_result = run_controlnet_openpose(img_path, anime_path)
                
                if openpose_result.image:
                    st.image(openpose_result.image, caption="OpenPose Skeleton")
                
                # --- Detailed OpenPose Data ---
                with st.expander("💀 OpenPose Keypoint Data (Coordinates)"):
                    for i, person in enumerate(openpose_result.people):
                        st.markdown(f"**Person {i}**")
                        
                        # Handle typos in lib/image/controlnet.py gracefully
                        left_hand = getattr(person, 'hand_left_keypoints_2d', None) or getattr(person, 'hand_left_keyopints_2d', None)
                        right_hand = getattr(person, 'hand_right_keypoints_2d', None) or getattr(person, 'hand_right_keyoints_2d', None)
                        
                        keypoint_data = {
                            "Body (Pose)": str(person.pose_keypoints_2d[:3]) + "..." if person.pose_keypoints_2d else "None",
                            "Face": str(person.face_keypoints_2d[:3]) + "..." if person.face_keypoints_2d else "None",
                            "Left Hand": str(left_hand[:3]) + "..." if left_hand else "None",
                            "Right Hand": str(right_hand[:3]) + "..." if right_hand else "None"
                        }
                        st.json(keypoint_data)
                # -------------------------------

                bboxes = controlnet2bboxes(openpose_result)
                
                base_img = Image.open(img_path)
                draw = ImageDraw.Draw(base_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 40) 
                except:
                    font = None 

                for i, box in enumerate(bboxes):
                    draw.rectangle(box, outline="red", width=5)
                    draw.text((box[0], box[1]), f"Char {i}", fill="red", font=font)
                st.image(base_img, caption="Detected Bounding Boxes")

            # --- Layout ---
            st.subheader("C. Layout & Bubbles")
            width, height = openpose_result.canvas_width, openpose_result.canvas_height
            layout = generate_layout(bboxes, panel_data, width, height)
            
            if layout is None:
                st.error("Failed to generate layout. (Character/Dialogue mismatch)")
            else:
                scored_layouts = similar_layouts(layout)
                
                if not scored_layouts:
                    st.error("No matching layouts found in database.")
                else:
                    best_match = scored_layouts[0]
                    ref_layout = best_match[0]
                    score = best_match[1]
                    pairs_iterator = best_match[2]
                    pairs_list = list(pairs_iterator)
                    
                    st.success(f"Best Match Found! Score: {score:.4f}")
                    
                    reconstructed_scored_layout = (ref_layout, score, pairs_list)
                    save_name_path = os.path.join(OUTPUT_DIR, f"panel{panel_idx}_final.png")
                    
                    generate_name(
                        openpose_result, 
                        layout, 
                        reconstructed_scored_layout, 
                        panel_data, 
                        save_name_path
                    )
                    
                    st.image(save_name_path, caption="✨ Final Result", use_column_width=True)

                    # --- Detailed Bubble & Mapping Data ---
                    st.divider()
                    with st.expander("🔍 Deep Dive: Coordinate Mapping & Bubble Locations"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("### 1. Generated Character BBoxes")
                            st.write("Format: `[min_x, min_y, max_x, max_y]`")
                            for idx, bbox in enumerate(bboxes):
                                st.code(f"Char {idx}: {bbox}")

                        with c2:
                            st.write("### 2. Template Bubble BBoxes")
                            st.write("Format: `[min_x, min_y, max_x, max_y]`")
                            
                            # Extract bubble coordinates from the template layout
                            for idx, element in enumerate(ref_layout.elements):
                                if isinstance(element, Speaker):
                                    speaker_label = f"Template Speaker {idx}"
                                    if element.text_info:
                                        # Show all bubbles for this speaker
                                        bubbles = [t["bbox"] for t in element.text_info]
                                        st.code(f"{speaker_label}: {bubbles}")
                                    else:
                                        st.code(f"{speaker_label}: No text info found")
                                elif isinstance(element, NonSpeaker):
                                    st.code(f"Template Element {idx}: Non-Speaker")

                        st.write("### 3. Final Assignment Map (The Intelligent Fix)")
                        st.caption("This shows exactly which of YOUR characters maps to which TEMPLATE bubble.")
                        
                        mapping_debug = []
                        for base_idx, ref_idx in pairs_list:
                            base_bbox = bboxes[base_idx]
                            ref_speaker = ref_layout.elements[ref_idx]
                            
                            # Re-simulate the logic to find WHICH bubble is used
                            target_bubble_bbox = "None"
                            if isinstance(ref_speaker, Speaker) and ref_speaker.text_info:
                                 bbox = [-1, -1, -1, -1]
                                 for text_obj in ref_speaker.text_info:
                                     if text_obj["bbox"][2] > bbox[2]:
                                         bbox = text_obj["bbox"]
                                 target_bubble_bbox = bbox
                            
                            mapping_debug.append({
                                "Your Character (Index)": base_idx,
                                "Mapped to Template (Index)": ref_idx,
                                "Target Bubble BBox": target_bubble_bbox
                            })
                        st.table(mapping_debug)