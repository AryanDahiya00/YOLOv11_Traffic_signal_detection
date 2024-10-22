# Traffic Signal Detection using YOLOv11

## Project Overview
This project implements a real-time traffic signal detection system using the YOLOv11 (You Only Look Once) architecture. The system efficiently identifies and classifies traffic signals into three distinct states: Go, Stop, and Warning.

## Dataset Information
- **Source**: [Traffic Signal Bulb Dataset](https://www.kaggle.com/datasets/doitdifferent/bulb-ts)
- **Classes**: 
  1. Go (Green Signal)
  2. Stop (Red Signal)
  3. Warning (Yellow/Amber Signal)


## Additional Test Video:
  - Source: [Traffic Signal Footage](https://youtu.be/iS5sq9IELEo?si=XFx0AVWQN5MTtQ86)
  - Usage: Small portion extracted for model testing
  - Purpose: Real-world performance validation
  - Note: Video used solely for educational and testing purposes with appropriate attribution

### Dataset Structure
```
dataset/
└── tt2/
    ├── test/
    │   ├── img001.jpg
    │   ├── img001.txt
    │   └── ...
    ├── train/
    │   ├── img101.jpg
    │   ├── img101.txt
    │   └── ...
    └── config.yaml 
```

### Annotation Format
Each image in the dataset is accompanied by a corresponding .txt file containing annotation information:
- Format: `<class_id> <x_center> <y_center> <width> <height>`
- Class IDs:
  - 0: Go
  - 1: Stop
  - 2: Warning
- All coordinates are normalized between 0 and 1
- Bounding box is defined by:
  - Center coordinates (x_center, y_center)
  - Dimensions (width, height)

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
- Base Model: YOLOv11n
- Input Resolution: 640x640

### Training Configuration
```yaml
model_params:
  architecture: YOLOv11
  input_size: 640
  anchors: 3
  classes: 3
```

### Data Preprocessing
The dataset underwent several preprocessing steps before training:

1. **Data Source**: Original dataset was sourced from the [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)

2. **Preprocessing Steps**:
   - Conversion of YAML annotations to YOLOv11-compatible format
   - Cross-checking of images and their corresponding annotations
   - Class consolidation: Compressed 6 original classes into 3 main categories
   - Data cleaning: Removed images and annotations with no objects
   - Class imbalance handling through strategic compression

## Model Training Pipeline
1. **Data Split**: 
   - Training set: 80%
   - Validation set: 20%

2. **Training Approach**:
   - Utilized transfer learning with COCO pretrained weights
   - Implemented custom training for traffic signal detection
   - Evaluated model performance using mAP (mean Average Precision)

## Testing Methodology
1. **Dataset Testing**:
   - Evaluation on held-out test set from original dataset
   - Performance metrics calculation on static images

2. **Real-world Testing**:
   - Additional validation using extracted video segments
   - Source: YouTube video clip ([Link](https://youtu.be/iS5sq9IELEo?si=XFx0AVWQN5MTtQ86))
   - Testing scenarios:
     - Various lighting conditions
     - Different traffic signal configurations
     - Real-world traffic scenarios

## Performance Metrics
Detailed performance metrics and evaluation results can be found in the 'results' folder of the project repository.

## Acknowledgments
- Original training dataset:  [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)
- Test video footage: Partial clip from [Traffic Signal Video](https://youtu.be/iS5sq9IELEo?si=XFx0AVWQN5MTtQ86)
  
## Notes
- The dataset provided includes preprocessed images ready for training
- Performance metrics and detailed evaluation results are documented in the results folder
- The model uses a simplified three-class system for efficient traffic signal detection

This implementation focuses on practical application while maintaining high accuracy in traffic signal detection. The preprocessing pipeline ensures optimal data quality for training, while the model architecture is optimized for real-time performance.
