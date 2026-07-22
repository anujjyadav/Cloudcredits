#  CNN Image Classification — Cats vs. Dogs

A complete end-to-end deep learning project for binary image classification using Convolutional Neural Networks (CNN) and Transfer Learning with MobileNetV2 on the Asirra Cats vs. Dogs dataset.

---

##  Project Structure

```
Image Classification/
│
├── Asirra_ cat vs dogs/          # Raw dataset (cat.*.jpg, dog.*.jpg)
│
├── image_classification.ipynb  # Main Jupyter Notebook (10 steps)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── project_report.md               # Full technical project report
```

---

##  Dataset

- **Source**: Asirra (Animal Species Image Recognition for Restricting Access) dataset
- **Classes**: Cat (0), Dog (1)
- **Total Images Used**: 1,000 (500 per class)
- **Format**: JPEG images
- **Split**: 70% Train / 15% Validation / 15% Test

---

##  Project Workflow (10 Steps)

| Step | Description |
|------|-------------|
| 1 | **Problem Definition** — Binary classification (Cat vs. Dog) |
| 2 | **Data Collection & Preparation** — Load, filter, organise images |
| 3 | **Exploratory Data Analysis (EDA)** — Visualise class distribution, sample images, pixel stats |
| 4 | **Feature Engineering** — Resize to 224×224, normalise pixel values [0,1] |
| 5 | **Data Splitting** — Stratified 70/15/15 train/val/test split |
| 6 | **Model Selection** — Custom CNN + MobileNetV2 Transfer Learning |
| 7 | **Model Training** — Data augmentation, EarlyStopping, ReduceLROnPlateau |
| 8 | **Model Evaluation** — Accuracy/Loss curves, Confusion Matrix, ROC-AUC |
| 9 | **Model Improvement** — Fine-tuning MobileNetV2, hyperparameter comparison |
| 10 | **Deployment** — `predict_image()` function + visual predictions grid |

---

##  Setup & Installation

### Prerequisites
- Python 3.9 or 3.10
- pip

### 1. Clone / Navigate to Project Directory
```bash
cd "e:/cloud credits/Image Classification"
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook image_classification.ipynb
```

### 5. Run All Cells
Use **Kernel → Restart & Run All** to execute the full notebook end-to-end.

---

##  Models Used

### Custom CNN
| Layer | Details |
|-------|---------|
| Conv2D (32 filters) | 3×3, ReLU, BatchNorm, MaxPool |
| Conv2D (64 filters) | 3×3, ReLU, BatchNorm, MaxPool |
| Conv2D (128 filters) | 3×3, ReLU, BatchNorm, MaxPool |
| Dense (256) | ReLU + Dropout(0.5) |
| Dense (1) | Sigmoid (binary output) |

### Transfer Learning — MobileNetV2
- **Base**: MobileNetV2 pre-trained on ImageNet (frozen)
- **Head**: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- **Fine-tuning**: Top 30 layers unfrozen in Step 9

---

## Expected Results

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|---------|
| Custom CNN | ~80% | ~78% | ~77% |
| MobileNetV2 (frozen) | ~95% | ~90% | ~88% |
| MobileNetV2 (fine-tuned) | ~97% | ~93% | ~91% |

---

##  Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow/Keras | 2.13.0 | Model building & training |
| NumPy | 1.24.3 | Numerical operations |
| Pandas | 2.0.3 | Data management |
| Matplotlib | 3.7.2 | Plotting & visualisation |
| Seaborn | 0.12.2 | Statistical visualisations |
| scikit-learn | 1.3.0 | Metrics, splitting |
| Pillow | 10.0.0 | Image I/O |
| OpenCV | 4.8.0.76 | Image processing |

---

##  License

This project is for educational purposes. The Asirra dataset was originally created by Microsoft Research.

---

##  Author

CNN Image Classification Project — Cats vs. Dogs  
Built with TensorFlow / Keras and Transfer Learning (MobileNetV2)
