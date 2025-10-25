# Sweet Spot: A Web Application to Classify Levels of Banana Ripeness using Convolutional Neural Networks

A project made for the completion of the course, CS 3310 - Modelling and Simulation, made by the authors Gigawin, Dave Shanna, Maclang, Waken Cean C., and Tagle, Allan C.

## Goal:

This project is geared towards making a web application for classifying banana ripeness using image-based classification methods. We aim to compare 3 different models, namely:

-   Self-designed CNN (VGG-conservative approach)

-   Transfer Learning

    -   ResNet101

    -   VGG9

## File Structure

The project is expected to have the following file structure:

```{txt}
MaS LE
├── CNN Related Code
│   ├── CNN_conservative.py
│   ├── CNN_ResNet101.py
│   ├── CNN_vanilla.py
│   └── import_data.py
├── Dataset Annotation Code
│   ├── Resize.py
│   └── Augment.py
├── Datasets
│   ├── Final Dataset
│   │   ├── train
│   │   ├── test
│   │   └── validation
│   ├── Real World Dataset
│   │   └── Banana Images
│   ├── Annotated Dataset
│   │   ├── Merged Dataset
│   │   │   ├── Rotten
│   │   │   ├── Ripe
│   │   │   ├── Overripe
│   │   │   └── Unripe
│   │   ├── Fayoum Uni Annotated Dataset
│   │   │   ├── Augment_List.csv
│   │   │   ├── Ripe
│   │   │   ├── Overripe
│   │   │   └── Unripe
│   │   ├── Fayoum Uni Dataset
│   │   │   ├── Ripe
│   │   │   ├── Overripe
│   │   │   └── Unripe
│   │   └── Shariar Dataset
│   │       ├── Rotten
│   │       ├── Ripe
│   │       ├── Overripe
│   │       └── Unripe
│   └── Original Datasets
│       ├── Shariar Dataset
│       │   ├── Rotten
│       │   ├── Ripe
│       │   ├── Overripe
│       │   └── Unripe
│       └── Fayoum Uni Dataset
│           ├── Unripe
│           ├── Ripe
│           └── Overripe
├── Conceptual Paper
│   └── GIGAWIN-MACLANG-TAGLE_Modelling and Simulation_Conceptual Paper.pdf
├── Literature
├── file_structure.txt
└── requirements.txt
```

## Setting up the Environment

The environment we are using will be focused on Machine Learning and Image Annotation.\
Do note that we will be using **Python 3.13.1**

### 1. Downloading the Python Version

Kindly head to the link: [Python 3.13.1 Download Release](https://www.python.org/downloads/release/python-3131/)

### 2. Downloading the file dependencies

Kindly run the following code in your terminal to download the required dependencies

```{python}
pip install -r requirements.txt
```

### 3. Downloading the Datasets

To download the specific datasets for you to manually augment, kindly head to the "Dataset Details.md" else, you can head to this specific drive link:

[GiMaTag Dataset](https://drive.google.com/drive/folders/1KPevtDtvTLYjyuAa00NRWERGVLXU0HNQ?usp=drive_link)