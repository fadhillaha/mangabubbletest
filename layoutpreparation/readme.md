# Manga109 Style Layout Trainer

This script analyzes a dataset of manga pages (specifically the Manga109 dataset with segmentation masks) to learn the statistical properties of manga layouts. It extracts three key models used for procedural manga generation:

1.  **Structure Model:** The probability of horizontal vs. vertical panel splits.
2.  **Importance Model:** The visual weight (area) of panels based on their rank on the page.
3.  **Shape Model:** The geometric variance (wobble/angles) of panel corners compared to perfect rectangles.

## Prerequisites

### A. Python Dependencies
Install the required libraries using pip:

```bash
pip install numpy opencv-python pycocotools tqdm
```
### B. Dataset
This script is designed to work with the Manga109 dataset and the MS92/MangaSegmentation dataset hosted on Hugging Face, which provides detailed panel segmentation masks in COCO format.

Download the Dataset:

Main Dataset: [Manga109](http://www.manga109.org/en/download.html) (Requires permission)

Segmentation Masks: [MangaSegmentation](https://huggingface.co/datasets/MS92/MangaSegmentation/tree/main) 
 (Download the jsons folder)

Prepare the Folder Structure: Place the downloaded jsons folder inside your project directory:


```plaintext
dataset/
├── Manga109/              # (Optional) Original Manga109 images
└── MangaSegmentation/
    └── jsons/             
        ├── AisazuNihairarenai.json
        ├── ...
```        

## Usage
Run the script from the terminal, providing the path to the folder containing the JSON files.

```bash
python layoutpreparation/train_manga109style.py "path/to/your/json_folder"
```

## Output
The script will generate a file named `style_models.json` in the same directory as the script.

This JSON file contains the learned probabilities and is used by the main pipeline (`src/pipeline.py`) to generate authentic manga layouts.