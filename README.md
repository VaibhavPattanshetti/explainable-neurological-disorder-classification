# explainable-neurological-disorder-classification
Multi-class deep learning system for MRI-based neurological disorder classification with a focus on explainable AI (XAI) to improve model transparency and clinical trust.

Problem:
Accurate diagnosis of neurological disorders from MRI scans is critical, yet challenging due to class imbalance and subtle visual differences between diseases.

This project presents a multi-class deep learning model that classifies 8 neurological conditions using MRI images with high clinical relevance.

🔹 Disorders Classified

Glioma Brain Tumor

Meningioma Brain Tumor

Pituitary Brain Tumor

Alzheimer’s Dementia (Very Mild, Mild, Moderate)

Multiple Sclerosis

Normal Brain

🔹 Dataset

Source: Kaggle – Three Brain Neurological Classes MRI Scans

Total Classes: 8

Highly imbalanced dataset handled using:

Class weights

Categorical focal loss

🔹 Model Architecture

Base Model: ResNet50 (ImageNet pre-trained)

Strategy:

Transfer Learning

Two-stage training:

Frozen backbone

Fine-tuning last 35 layers

Input size: 224 × 224 × 3

🔹 Techniques Used (THIS IMPRESSES RECRUITERS)

Custom Categorical Focal Loss for class imbalance

Data Augmentation (rotation, zoom, contrast, flip)

Class weighting

Early stopping & learning rate scheduling

AUC-based model checkpointing

🔹 Performance (Validation)
Metric	Value
Accuracy	91.99%
AUC (macro)	0.9952

Why AUC matters:
In medical diagnosis, recall and AUC are more critical than raw accuracy to reduce false negatives.

🔹 Class-wise Performance (Highlight This)

Glioma Recall: 0.97

Pituitary Tumor Recall: 1.00

Multiple Sclerosis Recall: 0.95

Normal Recall: 0.88

High recall for disease classes ensures fewer missed diagnoses.

🔹 Confusion Matrix

(Insert saved confusion matrix image here)

🔹 Inference Example
label, confidence = predict_image(model, image_path)
print(label, confidence)


Output:

Predicted: Normal
Confidence: 0.59

🔹 Tech Stack

Python

TensorFlow / Keras

NumPy, OpenCV, Matplotlib

Scikit-learn
