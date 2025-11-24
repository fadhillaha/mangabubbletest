generate_prompt_prompt = """
You are a professional prompt engineer for Stable Diffusion 1.5.
Your job is to generate a **CONTENT-ONLY** prompt description based on the script.

<Task>
Convert the script into a comma-separated list of Danbooru-style tags.
Focus ONLY on characters, actions, and setting.
</Task>

<Guidelines>
1.  **CONTENT ONLY:** Describe the characters (1boy, 1girl), clothes, emotions, pose, and background.
2.  **NO COLORS:** Convert colors to tones (e.g., "blonde" -> "light hair", "blue" -> "dark").
3.  **STRUCTURE:** `[Character 1] BREAK [Character 2] BREAK [Background]`
4.  **OUTPUT FORMAT:** - The output MUST BE only string.
- DO NOT answer in markdown format
- MUST answer in the row text.
</Guidelines>

Here is the input:
Descriptions: {descriptions}
Monologues: {monologues}
Dialogues: {dialogues}
{additional_guideines}

<Output Format>
- The output MUST BE only string.
- DO NOT answer in markdown format
- MUST answer in the row text.
</Output Format>
"""

enhancement_prompt = """
You are a professional prompt engineer.
Your job is to prepare the prompt for the **First Pass (Clean Generation)**.

<Task>
Take the content tags and PREPEND "High Quality" tags to ensure the image is clean and detailed (so OpenPose can detect it).
</Task>

<Guidelines>
1.  **PREPEND:** `masterpiece, best quality, highres, clean lines, sharp focus, anime style, monochrome, grayscale`
2.  **MAINTAIN CONTENT:** Keep the character and setting tags exactly as they are.
3.  **NO COLORS:** Ensure no color tags are added.
</Guidelines>

<Output Format>
- The output MUST BE only string.
- DO NOT answer in markdown format
- MUST answer in the row text.
</Output Format>
"""