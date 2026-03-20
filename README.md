# 🧠 NeuroScan AI — Explainable Neurological Disorder Classification

A **deep learning-based web application** that classifies **8 neurological disorders** from brain MRI scans using **ResNet50 Transfer Learning**, with full **Explainable AI (XAI)** support via **Grad-CAM** and **LIME** to make predictions transparent and clinically trustworthy.

---

## 🔗 Model Download

The trained model is hosted on Hugging Face due to its large size (217MB).

👉 [Download Model from Hugging Face](https://huggingface.co/Vaibhav182005/neuroscan-resnet50/blob/main/resenet_50_new_dataset_92_v2.keras)

After downloading, place `resenet_50_new_dataset_92_v2.keras` in the same folder as `app.py` before running the app.

---

## 📖 Project Description

Early and accurate detection of neurological disorders from MRI scans is critical for timely treatment. This project leverages deep learning to:

- **Classify brain MRI scans** into 8 neurological categories automatically
- **Explain predictions** visually using Grad-CAM and LIME so doctors can trust the model
- **Highlight affected brain regions** to support clinical decision-making
- **Provide confidence scores** for every prediction

This web app accepts a brain MRI image, runs it through a fine-tuned ResNet50 model, and returns the predicted condition along with visual explanations of what the model focused on.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Model | ResNet50 (Transfer Learning + Fine-tuning) |
| Validation Accuracy | **92%** |
| Validation AUC | **0.9952** |
| Train Accuracy | 96% |
| Train AUC | 0.9985 |
| Total Dataset Size | 24,588 MRI images |
| Number of Classes | 8 |

---

## 🏥 Detectable Conditions

| # | Condition | Category |
|---|---|---|
| 1 | 1st Brain Tumor Glioma | Brain Tumor |
| 2 | 2nd Brain Tumor Meningioma | Brain Tumor |
| 3 | 3rd Brain Tumor Pituitary | Brain Tumor |
| 4 | Alzheimer's Dementia 1st Very Mild | Alzheimer's |
| 5 | Alzheimer's Dementia 2nd Mild | Alzheimer's |
| 6 | Alzheimer's Dementia 3rd Moderate | Alzheimer's |
| 7 | Multiple Sclerosis | MS |
| 8 | Normal | Normal |

---

## 🗃️ Dataset

The model was trained on the **Three Brain Neurological Classes MRI Scans** dataset from Kaggle.

🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/omarradi/three-brain-neurological-classes-mri-scans)

| Class | Images |
|---|---|
| Normal | 8,104 |
| 3rd Brain Tumor Pituitary | 3,490 |
| Multiple Sclerosis | 3,195 |
| 2nd Brain Tumor Meningioma | 3,160 |
| 1st Brain Tumor Glioma | 3,000 |
| Alzheimer's Dementia 1st Very Mild | 2,240 |
| Alzheimer's Dementia 2nd Mild | 896 |
| Alzheimer's Dementia 3rd Moderate | 503 |

---

## 🤖 Model Architecture & Training

### Base Model — ResNet50
- Pretrained on ImageNet weights
- Input size: 224×224×3
- Custom classification head added on top

### Training Strategy — 2 Stage Approach

**Stage 1 — Feature Extraction (20 epochs)**
- Base ResNet50 layers frozen
- Only classification head trained
- Learning rate: 1e-4
- Best Val AUC: 0.9839

**Stage 2 — Fine Tuning (12 epochs)**
- Last 35 layers of ResNet50 unfrozen
- Learning rate: 1e-5
- Best Val AUC: 0.9952 ✅

### Key Training Techniques
- **Focal Loss** (γ=2.5) — handles class imbalance effectively
- **Class Weights** — computed via sklearn for balanced training
- **Data Augmentation** — random flip, rotation, zoom, contrast
- **ReduceLROnPlateau** — adaptive learning rate scheduling
- **EarlyStopping** — prevents overfitting

---

## 🔍 Explainable AI (XAI) Methods

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Uses gradients flowing back through the CNN to highlight influential spatial regions
- **Red/warm areas** = regions the model focused on most
- **Blue/cool areas** = regions with little influence
- For tumors, hot-spots align with the lesion location
- For Alzheimer's, attention falls on ventricles and hippocampal region

### LIME (Local Interpretable Model-agnostic Explanations)
- Perturbs superpixel regions to find which parts most affect the prediction
- **Yellow outlines** mark the most important superpixel regions
- **Heatmap overlay** shows intensity of each region's contribution
- Model-agnostic — works independently of the neural network architecture

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Base Model | ResNet50 (ImageNet pretrained) |
| XAI — Grad-CAM | TensorFlow GradientTape |
| XAI — LIME | lime library |
| Image Processing | Pillow, scikit-image |
| Visualization | Matplotlib |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/VaibhavPattanshetti/explainable-neurological-disorder-classification.git
cd explainable-neurological-disorder-classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the model**

👉 [Download from Hugging Face](https://huggingface.co/Vaibhav182005/neuroscan-resnet50/blob/main/resenet_50_new_dataset_92_v2.keras)

Place `resenet_50_new_dataset_92_v2.keras` in the same folder as `app.py`.

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open in browser**

Go to `http://localhost:8501`

---

## 📁 Project Structure

```
explainable-neurological-disorder-classification/
├── app.py                                  # Main Streamlit application
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore file
├── .gitattributes                          # Git configuration
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation

Model hosted separately on Hugging Face:
└── resenet_50_new_dataset_92_v2.keras      # Trained ResNet50 model (217MB)
```

---

## 🖥️ How to Use the App

1. **Upload** a brain MRI scan (JPG or PNG)
2. **Toggle** Grad-CAM and/or LIME on or off as needed
3. **Adjust** LIME sample count (more samples = better accuracy but slower)
4. Click **Analyze Scan**
5. View the **predicted condition** with confidence score
6. Explore the **Grad-CAM heatmap** to see where the model focused
7. Explore the **LIME explanation** to see important superpixel regions
8. **Download** the explanation images if needed

---

## ⚠️ Disclaimer

> This tool is intended for **research and educational purposes only**.
> It is **not a certified medical device** and should **not be used for clinical diagnosis**.
> Always consult a qualified medical professional for diagnosis and treatment.

---
