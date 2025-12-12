# NameGen

> ### **Attribution**
>
> This project is modified fork of **NameGen** by **Kazuki Kitano** \
> Original Author : **Kazuki Kitano** \
> Original Repository : **https://github.com/kitano-kazuki/bubbleAlloc/tree/pipeline** \
> The **modifications** in this repository were implemented as part of an internship project at **Parsola Inc.**


## Overview
**NameGen** is a tool to generate a manga **"Name"** (rough storyboard) from a text script. It constructs a speech bubble database using the **Manga109**  datasets, which serves as a reference for the automatic placement of speech bubbles.

The following features were added to the original repository:

1.  **Web Interface :**
    * Added a Streamlit user interface to put input script, generate, and view resulting generated manga pages without using command-line.

2.  **Expanded LLM Support :**
    * Integrated Google Gemini model as an alternative to the original OpenAI dependency.
    * This allows users to choose between models for script analysis and panel breakdown.

3.  **Automatic Scoring System :**
    * Added evaluation metric to select the best panel variations.
    * **CLIP Score :** Measures how well the generated image matches the script prompt using CLIP model.
    * **Geometric Penalty :** Detects and penalizes speech bubbles that overlap with character's faces or bodies.

4.  **Layout Generation & Page Compositing :**
    * **Algorithmic Layout :** Implemented the "Automatic Stylistic Manga Layout" technique ([Cao et al., 2012](https://www.ying-cao.com/projects/stylistic_layout/manga_layout.htm)) to generate dynamic panel structures based on the Manga109 dataset.
    * **Page Compositor :** Added compositor that arrange panels and layouts into pages, handling panel resizing, aspect ratio fitting, and border drawing.

## 🛠️ Installation

### 1. Prerequisites
Clone the repository and install the dependencies.

```bash
# Clone this repository
git clone https://github.com/fadhillaha/mangabubbletest.git
cd mangabubbletest

# 1. Install Project Dependencies
pip install -r requirements.txt

# 2. Install the package in editable mode
pip install -e .
```

### 2. Manga Database Construction
NameGen requires a dataset from the Manga109 and its additional annotation, Manga109Dialog to construct database of manga speech bubble.

#### Required Data:

- [Manga109](http://www.manga109.org/en/download.html)
- [Manga109Dialog](https://github.com/manga109/public-annotations)


#### Setup Steps:

Download the datasets and place them in the `dataset` folder as shown below:

```plaintext
- dataset/
  |- Manga109/
     |- annotations/
     |- images/
     |- books.txt
     |- readme.txt
  |- Manga109Dialog/
     |- AisazuNihairarenai.xml
     |- ...
```
Run the curation script to extract bubble data:
```bash
python util/curate_dataset.py
```
Combine the annotations into a single database file.
```bash
python src/dataprepare.py
```
### (Optional) Panel Layout Training
The project comes with a pre-trained layout model (`layoutpreparation/style_models_manga109.json`).This step is only needed to recreate the layout model (e.g., to learn new panel arrangements).

#### Requirement : 
- [MangaSegmentation](https://huggingface.co/datasets/MS92/MangaSegmentation/tree/main) 

#### Training Steps:

Place the segmentation JSONs in `dataset/MangaSegmentation/jsons/`.

Run the training script:

```bash
python layoutpreparation/train_manga109style.py dataset/MangaSegmentation/jsons/
```

This will generate a new `style_models.json` file based on the provided annotations.

### 3. Stable Diffusion Setup
This project uses Stable Diffusion WebUI for image generation via API.

Install [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) in a separate directory.

Edit the webui-user.sh (Linux/Mac) or webui-user.bat (Windows) file to enable the API:

```bash
export COMMANDLINE_ARGS="--api --xformers"
```
Models:

Model: [T-anime-v4](https://civitai.com/models/69552/t-anime-v4-pruned) (Place in models/Stable-diffusion)

LoRA: [Pose Sketches](https://civitai.com/models/1314152?modelVersionId=1483420) (Place in models/Lora)

Launch the WebUI. Ensure http://127.0.0.1:7860/docs is accessible.


### 4. API Configuration
Create a .env file in the root directory for LLM API access (OpenAI or Google LLM model):

```plaintext
API_KEY=your__api_key_here
```


## 💻 Usage
### Option A: Web Interface
Use the web dashboard.
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

Enter your script text and click Generate Panels.

### Option B: Command Line 
Generate directly from a text file.

```bash
python src/pipeline.py --script_path examples/your_script.txt --output_path path/to/output_dir
```

After running the pipeline, the results will be in the output folder.

