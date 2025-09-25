# Tuberculosis Detection from Chest X-ray Images using Deep Learning

[![Research Project](https://img.shields.io/badge/Project-Research-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-blue)](#)
[![License](https://img.shields.io/badge/License-Research-blue)](#)

---

## Live Project Link
 https://tb-detection-deeplearning-2.onrender.com

## Project Overview

This project presents an advanced AI-powered system for detecting tuberculosis (TB) using chest X-ray images. The system leverages state-of-the-art deep learning architectures to provide a two-step analysis:
- First, **lung segmentation** is performed with a U-Net based model to isolate lung regions and reduce noise.
- Then, **TB classification** is performed on these segmented regions using a DenseNet121-based classifier.

The goal is to develop an accurate, explainable, and research-focused AI tool for tuberculosis detection, designed for educational and academic use.

---

## Project Structure

- `preprocessing.py` — Data loading, cleaning, resizing, normalization, and dataset splitting for both segmentation and classification.
- `segmentation.py` — U-Net model definition, training with data augmentation, cross-validation, and evaluation.
- `classification.py` — DenseNet121 model for TB detection, training with k-fold CV, early stopping, and evaluation.
- `train.py` — Training orchestration, integrating segmentation and classification training workflows.
- `utils.py` — Helper functions for visualization (including Grad-CAM), metrics, and data conversion.
- `webapp/` — Modern Flask-based web UI with image upload, prediction visualization, and interactive results display.

---

## Key Features

- **Two-stage approach:** Combines lung segmentation with disease classification for improved accuracy.
- **Deep learning architectures:** Utilizes U-Net for segmentation and DenseNet121 for classification.
- **Cross-validation:** Implements k-fold cross-validation for both model training and evaluation to ensure robustness.
- **Data augmentation:** Uses diverse augmentation techniques to enhance model generalization.
- **Explainability:** Integrates Grad-CAM visualizations to highlight important regions influencing the AI’s decisions.
- **Research disclaimer:** Clear indications that the system is for research, educational purposes only and not for clinical use.
- **Modern UI:** User-friendly web application for image upload, results display, and risk visualization.

---

## Installation

1. Clone the repository:

git clone https://github.com/atharvakanchan25/TB-Detection-DeepLearning.git

2. Create and activate a Python virtual environment:
    bash - python -m venv venv
    Windows:
    .\venv\Scripts\activate


Use the web interface to upload chest X-ray images and view TB detection results with visual explanations.

---

## Important Disclaimer

> **This project is strictly for research and educational purposes only.**  
> It is **NOT** approved for clinical or diagnostic use.  
> Always consult qualified healthcare professionals for medical advice and diagnosis.  
> This AI system may produce false positives or false negatives and should not replace professional judgment.

---

## Technologies

- Python 3.13  
- TensorFlow/Keras  
- Flask Web Framework  
- OpenCV and Pillow for image processing  
- NumPy, Matplotlib for data handling and visualization  
- Scikit-learn for evaluation and CV  
- Bootstrap 5 for frontend UI

---

## Contribution

This is an academic research project. Pull requests and issues for improvements are welcome. Please ensure contributions align with research integrity and ethical AI use.

---

## License

This project is provided for research purposes without warranty. Please read [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaborations, please contact:

Atharva Kanchan  
Email: [athatvakanchan959@gmail.com]  
GitHub: [https://github.com/atharvakanchan25]
