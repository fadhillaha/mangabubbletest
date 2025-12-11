This README.md explains how to use the train_manga109style.py script to train a layout style model using the Manga109 segmentation dataset.

Manga109 Style Layout Trainer
This script analyzes a dataset of manga pages (specifically the Manga109 dataset with segmentation masks) to learn the statistical properties of manga layouts. It extracts three key models used for procedural manga generation:

Structure Model: The probability of horizontal vs. vertical panel splits.

Importance Model: The visual weight (area) of panels based on their rank on the page.

Shape Model: The geometric variance (wobble/angles) of hand-drawn panel corners compared to perfect rectangles.

1. Prerequisites
A. Python Dependencies
Install the required libraries using pip:
pip install numpy opencv-python pycocotools tqdm

pycocotools: Required to decode the RLE (Run-Length Encoding) segmentation masks used in the dataset.

opencv-python: Used for contour detection and polygon approximation.

B. Dataset
This script is designed to work with the Manga109 dataset and MS92/MangaSegmentation dataset hosted on Hugging Face, which provides detailed panel segmentation masks for the Manga109 dataset in COCO format.

Download the Dataset:
Main dataset : http://www.manga109.org/en/
Segmentation Dataset https://huggingface.co/datasets/MS92/MangaSegmentation/tree/main

Prepare the Folder:
dataset/
├── Manga109/                 # (Optional) Original Manga109 dataset
│   ├── Annotation/
│   └── Image/
└── Manga109Segmentation/     # Place downloaded HuggingFace data here
    └── json_folder/


2. Usage
Run the script from the terminal, providing the path to the folder containing your JSON files.

Bash

python train_manga109style.py "path/to/your/json_folder"

Example:

python train_manga109style.py "./dataset/Manga109Segmentation/json_folder/"