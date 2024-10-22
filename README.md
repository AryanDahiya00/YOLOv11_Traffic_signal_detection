# Traffic Signal Detection using YOLOv11

## Project Overview
This project implements a traffic signal detection system using YOLOv11 (You Only Look Once) architecture to detect and classify traffic signals in real-time. The system can identify three distinct states of traffic signals: Go, Stop, and Warning.

## Dataset Information
- **Source**: [Traffic Signal Bulb Dataset](https://www.kaggle.com/datasets/doitdifferent/bulb-ts)
- **Classes**: 
  1. Go (Green Signal)
  2. Stop (Red Signal)
  3. Warning (Yellow/Amber Signal)

### Dataset Structure
```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
└── test/
    ├── images/
    │   ├── img101.jpg
    │   ├── img102.jpg
    │   └── ...
    └── labels/
        ├── img101.txt
        ├── img102.txt
        └── ...
```

### Annotation Format
- Each image has a corresponding .txt file containing annotations
- Annotation format: `<class_id> <x_center> <y_center> <width> <height>`
  - class_id: 0 for Go, 1 for Stop, 2 for Warning
  - All coordinates are normalized (0-1)
  - (x_center, y_center) represents the center of the bounding box
  - (width, height) represents the dimensions of the bounding box

## Technical Implementation

### Dependencies
```
- Python 3.8+
- PyTorch
- YOLOv11
- OpenCV
- NumPy
- Pandas
```

### Model Architecture
- Base: YOLOv11
- Input size: 640x640
- Backbone: CSPDarknet
- Neck: PANet
- Head: YOLOv11 detection head

### Training Configuration
```yaml
model_params:
  architecture: YOLOv11
  input_size: 640
  anchors: 3
  classes: 3

training_params:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
  optimizer: Adam
  momentum: 0.937
  weight_decay: 0.0005
```

### Data Preprocessing
1. Image resizing to 640x640
2. Data augmentation techniques:
   - Random horizontal flip
   - Random rotation (±15 degrees)
   - Random brightness and contrast
   - Mosaic augmentation

## Model Training Process
1. Dataset splitting: 80% training, 20% validation
2. Transfer learning from COCO pretrained weights
3. Custom training on traffic signal dataset
4. Learning rate scheduling with cosine annealing
5. Model evaluation using mAP (mean Average Precision)

## Inference Pipeline
1. Input image/video stream
2. Preprocessing (resize to 640x640)
3. Model inference
4. Post-processing:
   - Non-maximum suppression (NMS)
   - Confidence thresholding (default: 0.25)
   - IoU thresholding (default: 0.45)
5. Visualization of results

## Performance Metrics
- mAP@0.5: To be updated after training
- FPS: To be updated after optimization
- Inference time: To be updated after testing

## Future Improvements
1. Implementation of TTA (Test Time Augmentation)
2. Model quantization for faster inference
3. Integration with traffic management systems
4. Support for night-time detection
5. Extension to more traffic signal types

## Project Structure
```
project/
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── weights/
│   └── configs/
├── src/
│   ├── train.py
│   ├── detect.py
│   ├── utils/
│   └── dataset.py
├── notebooks/
│   └── exploratory_analysis.ipynb
└── requirements.txt
```

## Usage Instructions
(To be added after implementation)

## References
1. YOLOv11 Paper (Add reference when available)
2. Traffic Signal Dataset: https://www.kaggle.com/datasets/doitdifferent/bulb-ts
3. Related research papers and implementations
