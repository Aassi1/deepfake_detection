Assi Assi

## Development Timeline Note

Due to personal circumstances, I started this project on November 22
rather than November 1 as originally planned. The compressed timeline
required intensive daily work (6-8 hours/day) to complete all components.

My commit history reflects the actual development timeline.

# Deepfake Detection System

A CNN-based deepfake video detector built for COMP 3106 (Intro to AI) at Carleton University.

Uses transfer learning with EfficientNet-B0 to detect deepfakes with **88.84% accuracy** on the Celeb-DF v2 dataset.

---

## What This Does

Takes a video → extracts faces → classifies each as real or fake → gives overall verdict.

**Key results:**

- 88.84% test accuracy
- 96.73% precision on fake detection

---

## Quick Start

### Install

```bash

git clone https://github.com/2025F-COMP3106/project-group-194.git

conda create -n deepfake_env python=3.10
conda activate deepfake_env

pip install -r requirements.txt
```

### Run Web Interface

```bash
streamlit run src/demo.py
```

Upload a video and get results with confidence scores.

---

## How It Works

**3-stage pipeline:**

1. **Face Extraction** - Sample frames at 5 FPS, detect faces with MediaPipe
2. **Classification** - EfficientNet-B0 CNN predicts real/fake for each face
3. **Aggregation** - Majority vote across frames for final verdict

**Training approach:**

- Used pretrained EfficientNet-B0 (ImageNet weights)
- Froze backbone, trained only classification head (656K parameters)
- Class-weighted loss to handle 10:1 fake-to-real imbalance

---

## Dataset

**Celeb-DF v2**: 6,229 videos (590 real, 5,639 fake)

Extracted **391,691 face images** from videos:

- 37,614 real faces
- 354,077 fake faces

Split: 70% train / 15% validation / 15% test

Download from: https://github.com/yuezunli/celeb-deepfakeforensics

---

## Training

```bash
python src/extract_faces.py  # Extract faces from videos (~1.5 hours)
python src/train.py          # Train model (~90 minutes on RTX 3050)
```

**Config:**

- Batch size: 32
- Learning rate: 0.0001
- Optimizer: Adam
- Epochs: 20 with early stopping

---

## Evaluation

```bash
python src/evaluate.py
```

Generates confusion matrix, ROC curve, and metrics.

**Results on test set (58,755 faces):**

| Metric    | Real   | Fake   |
| --------- | ------ | ------ |
| Precision | 44.89% | 96.73% |
| Recall    | 71.10% | 90.73% |

**Confusion Matrix:**

```
              Predicted
            Real    Fake
Real        4,012   1,631
Fake        4,926   48,186
```

---

## Project Structure

```
src/
├── model.py           # EfficientNet model
├── extract_faces.py   # Face extraction
├── train.py           # Training script
├── evaluate.py        # Evaluation
└── demo.py            # Streamlit interface

data/
├── videos/            # Raw videos
└── faces/             # Extracted faces

models/
└── best_model.pth     # Trained checkpoint

results/
├── confusion_matrix.png
└── roc_curve.png
```

---

## Limitations

- **Trained on 2019-era deepfakes** - Newer methods (Sora, diffusion models) would likely fool it
- **Conservative bias** - 28.9% false positive rate on real faces due to data imbalance
- **No temporal analysis** - Only looks at individual frames, not frame-to-frame consistency

---

## What I Learned

- Transfer learning is powerful - training only 14% of parameters got good results
- Class imbalance is tricky - weighted loss helps but creates tradeoffs
- Real-world deployment needs more than accuracy - speed and explainability matter
- Deepfake detection is an arms race - models need constant updates

---

## Requirements

```
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
mediapipe==0.10.7
streamlit==1.28.1
numpy==1.24.3
pandas==2.1.3
matplotlib==3.8.2
scikit-learn==1.3.2
```

Full list in `requirements.txt`

---

## References

- Dataset: [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- Model: [EfficientNet](https://docs.pytorch.org/vision/main/models/efficientnet.html)
- Face Detection: [MediaPipe](https://google.github.io/mediapipe/)

---

## Course Info

**COMP 3106** - Introduction to Artificial Intelligence  
**Instructor:** Prof. Matthew Holden  
**Term:** Fall 2025  
**Student:** Assi Assi (101302142)

---

## License

MIT License - free to use for academic purposes
