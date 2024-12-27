# Traffic Sign and Signal Detection using YOLOv11

## Overview
This repository contains a comprehensive implementation of traffic sign and signal detection using the latest version of YOLOv11. The model is designed to accurately detect and classify various traffic signs and signals in real-time, making it suitable for autonomous driving systems and traffic monitoring applications.

## Repository Structure

### Data_Preprocessing/
Contains scripts and utilities for preprocessing the training and validation datasets:
- Data normalization scripts
- Image augmentation tools
- Dataset splitting utilities
- Annotation format conversion tools

### Full_Model_Results/
Contains the evaluation results and metrics of the combined model that detects both traffic signs and signals:
- Confusion matrices
- Precision-recall curves
- mAP (mean Average Precision) scores
- Training and validation loss graphs
- Performance benchmarks

### Model_Testing/
Collection of test videos demonstrating the model's performance:
- Urban scenarios
- Highway conditions
- Different weather conditions
- Various lighting conditions
- Sample detection outputs

### Traffic_Lights_Results/
Contains results from the initial model focused solely on traffic light detection:
- Performance metrics
- Detection accuracy reports
- Validation results
- Model weights and configurations

## Key Features

- Real-time detection capabilities
- Support for multiple traffic sign and signal classes
- Robust performance under various lighting conditions
- High accuracy and low false positive rate
- Optimized for embedded systems deployment

## Data Extraction

The `extract_info_xml.py` script is used to extract annotation data from XML files. This script:
1. Parses XML annotation files
2. Extracts bounding box coordinates
3. Converts annotations to YOLO format
4. Generates label files for training

## Model Architecture

The detection system employs YOLOv11 with the following specifications:
- Backbone: CSPDarknet
- Neck: PANet
- Head: Modified YOLOv11 detection head
- Custom anchors optimized for traffic signs and signals

## Training Pipeline

### Dataset Preparation
1. Data collection from various sources
2. Annotation in Pascal VOC format
3. Conversion to YOLO format
4. Dataset splitting (80% training, 20% validation)

### Training Configuration
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Number of epochs: 100
- Input resolution: 640x640

## Results

### Performance Metrics
- mAP@0.5: 94.2%
- Inference time: 15ms on RTX 3080
- FPS: 60+ on test hardware

### Detection Classes
1. Traffic Signs:
   - 73 classes

2. Traffic Signals:
   - Red light
   - Yellow light
   - Green light

## Requirements
- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- CUDA 11.0+ (for GPU support)
- Additional dependencies in requirements.txt

## Future Improvements
- Integration with tracking algorithms
- Support for night-time detection
- Mobile device optimization
- Extended class support
- Real-time edge device deployment


## Contact
For any queries or suggestions, please open an issue in the repository.
