import os
import json
import time
from lib.script.prompt import storyboard_prompt

def analyze_storyboard(client, panels, output_path, max_retry=3):
    """
    Asks the LLM to assign page numbers, importance, and aspect ratios to every panel.
    """
    # 1. Check if we already did this (Resume capability)
    metadata_path = os.path.join(output_path, "panel_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            print("Loaded existing storyboard metadata.")
            return json.load(f)

    # 2. Prepare the payload
    messages = [
        {"role": "system", "content": storyboard_prompt},
        {"role": "user", "content": json.dumps(panels, ensure_ascii=False)},
    ]

    print("Analyzing storyboard structure (Pages, Importance, Aspect Ratio)...")

    # 3. Call the LLM
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash", 
                messages=messages, 
                temperature=0.0
            )
            result_text = response.choices[0].message.content
            
            # Clean up potential markdown formatting from LLM (e.g. ```json ... ```)
            if result_text.startswith("```"):
                result_text = result_text.strip("`").replace("json\n", "").strip()

            metadata = json.loads(result_text)

            # 4. Save the result
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            
            return metadata

        except json.JSONDecodeError as e:
            print(f"JSON Error on attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(1)
            
    raise Exception("Failed to generate storyboard metadata after retries.")