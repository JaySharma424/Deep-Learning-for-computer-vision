# Deep-Learning-for-computer-vision
# 🙂 DeepFER — Facial Emotion Recognition Using Deep Learning

<p align="center">
  <b>A practical seven-class facial emotion recognition system powered by CNNs, ResNet50 Transfer Learning, and Streamlit.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-8A2BE2" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

---

## 📌 Project Overview

**DeepFER (Facial Emotion Recognition)** is a computer vision project that classifies human facial expressions into seven emotion categories using deep learning.

The project combines a lightweight preprocessing pipeline with a CNN baseline and a **ResNet50 transfer-learning model**. It also includes model evaluation, error analysis, Grad-CAM interpretability, and a **Streamlit interface** for interactive inference through image upload or camera capture.

The project is designed around a simple practical workflow:

> **Image → Preprocessing → Deep Learning Model → Emotion Prediction → Confidence → Streamlit Interface**

### 🎯 Emotion Classes

| Emotion       | Description                   |
| ------------- | ----------------------------- |
| 😠 `angry`    | Angry facial expression       |
| 🤢 `disgust`  | Disgusted facial expression   |
| 😨 `fear`     | Fearful facial expression     |
| 😊 `happy`    | Happy facial expression       |
| 😐 `neutral`  | Neutral / non-expressive face |
| 😢 `sad`      | Sad facial expression         |
| 😲 `surprise` | Surprised facial expression   |

---

## ✨ Key Highlights

* 🧠 **CNN baseline** for a straightforward deep-learning comparison
* 🚀 **ResNet50 Transfer Learning** as the main model
* 🖼️ Visual exploration with multiple images from every emotion class
* 🔍 Dataset integrity checks for missing, empty, and unreadable images
* 📊 Class-distribution analysis
* 🧹 Basic preprocessing without unnecessary complexity
* 🔄 Lightweight augmentation using rotation and horizontal flipping
* ⚖️ Class-weighted training to address class imbalance
* 📈 Epoch-by-epoch **Training vs Validation Accuracy** visualization
* 📉 Training and validation loss curves
* 🧾 Accuracy, precision, recall, and F1-score
* 🎯 Confusion matrix and misclassified-image analysis
* 🔥 Grad-CAM visualization for model interpretability
* 💾 Saved Keras model and class-label metadata
* 🌐 **Streamlit UI** with image upload and camera capture
* ☁️ Google Colab friendly training and demo workflow

---

## 🏗️ System Architecture

```text
                    ┌───────────────────────┐
                    │   Facial Image Input  │
                    │ Upload / Camera       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Basic Preprocessing   │
                    │ Resize + Rescaling    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Data Augmentation     │
                    │ Rotation + Flip       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     ResNet50 CNN      │
                    │   Transfer Learning   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Softmax Classifier  │
                    │       7 Classes       │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┼───────────────┐
                 ▼              ▼               ▼
          Emotion Label    Confidence      Top Predictions
                 │
                 ▼
        ┌───────────────────┐
        │    Streamlit UI   │
        │ Upload / Camera   │
        └───────────────────┘
```

---

## 📂 Dataset

The project uses a facial-expression dataset organized into seven emotion folders:

```text
train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/

validation/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

The notebook automatically searches the extracted dataset for the `train` and `validation` directories, so the ZIP does not need to follow one hard-coded extraction path.

### Dataset analysis included in the notebook

* Class counts for training and validation sets
* Missing-class checks
* Empty-file checks
* Unreadable/corrupt-image checks
* Representative images from all seven classes
* Multiple samples per class for visual comparison

---

## 🔧 Preprocessing & Augmentation

The project intentionally keeps preprocessing **basic and easy to explain**.

### Preprocessing

* Resize images to the model input size
* Convert to RGB for ResNet50 compatibility
* Rescale pixel values to the `[0, 1]` range

### Augmentation

Only lightweight augmentation is used:

* Small image rotation
* Horizontal flipping

This keeps the pipeline simple while introducing moderate visual variability during training.

---

## 🧠 Model Development

### 1. CNN Baseline

The notebook contains an optional CNN baseline to provide a simple reference model before transfer learning.

### 2. ResNet50 Transfer Learning

The main model uses **ResNet50** as the feature extractor.

The training strategy is divided into two stages:

**Stage 1 — Classifier Head Training**

* ResNet50 backbone initially frozen
* New classification layers trained for the seven emotion classes

**Stage 2 — Fine-Tuning**

* Only the final portion of the backbone is unfrozen
* A smaller learning rate is used
* Batch-normalization layers remain frozen for more stable fine-tuning

---

## 📈 Training Visualization

The notebook generates:

### Accuracy Curve

```text
Training Accuracy ────────────────╮
                                  │
Validation Accuracy ─────────────╯
                 Epoch →
```

The notebook plots **training accuracy and validation accuracy after every epoch**, including a marker showing where fine-tuning begins.

### Loss Curve

Training and validation loss are plotted across the complete training process to help identify convergence and possible overfitting.

---

## 📊 Model Evaluation

DeepFER evaluates the final model using:

| Metric           | Purpose                                 |
| ---------------- | --------------------------------------- |
| Accuracy         | Overall classification performance      |
| Precision        | How often predicted classes are correct |
| Recall           | How well each class is detected         |
| F1-score         | Balance between precision and recall    |
| Confusion Matrix | Class-wise prediction analysis          |

The notebook also displays representative **misclassified images**, including the true class, predicted class, and prediction confidence.

> **Note:** The README intentionally does not hard-code a final accuracy or F1-score. Run the notebook on the supplied dataset to generate the actual metrics for your training run.

---

## 🔥 Explainability with Grad-CAM

DeepFER includes an optional **Grad-CAM** step to visualize which areas of an input face contributed most strongly to the predicted emotion.

This helps answer:

> **“Which part of the image did the model focus on when making its prediction?”**

---

## 🌐 Streamlit Application

The project includes a user-facing Streamlit application for interactive inference.

### UI Features

* 📤 Upload a JPG/PNG face image
* 📷 Capture an image using the browser camera
* 🧠 Predict one of seven emotions
* 📊 Display prediction confidence
* 🥇 Show the top predicted classes
* ⚡ Reuse the saved trained model for inference

---

## 🚀 Getting Started in Google Colab

Open:

```text
DeepFER_Facial_Emotion_Recognition_Reference_Based.ipynb
```

in Google Colab.

Enable a GPU:

```text
Runtime → Change runtime type → GPU
```

Then upload the facial-emotion dataset ZIP and execute the notebook from top to bottom.

---

## 🛠️ Tech Stack

| Technology         | Role                        |
| ------------------ | --------------------------- |
| Python             | Project development         |
| TensorFlow / Keras | Deep learning and training  |
| ResNet50           | Transfer-learning backbone  |
| NumPy              | Numerical operations        |
| Pandas             | Dataset analysis            |
| Matplotlib         | Visualizations              |
| Scikit-learn       | Metrics and evaluation      |
| Pillow             | Image handling              |
| Streamlit          | Interactive web application |
| Google Colab       | Training environment        |

---

## 🌍 Potential Applications

* Human-computer interaction
* Interactive assistants
* Customer-service analytics
* User-experience research
* Educational or experimental interfaces
* Emotion-aware computer-vision prototypes

> Facial-expression classification should not be treated as a clinical diagnostic system or as a definitive measurement of a person's internal mental state.

---

## 🔮 Future Improvements

* Larger and more diverse datasets
* Hyperparameter optimization
* Better robustness to pose and lighting
* Face detection and alignment
* Model quantization for edge deployment
* Persistent cloud deployment
* Additional explainability and calibration analysis

---

## 👤 Author

**Dhananjay Kumar Sharma**

Master of Computer Applications (MCA) | Generative AI | Agentic AI | Data Science

GitHub: **JaySharma424**

---

## ⭐ Support

If you find the project useful, consider giving the repository a **star ⭐** and exploring the notebook to see the complete workflow from dataset analysis to an interactive Streamlit application.
