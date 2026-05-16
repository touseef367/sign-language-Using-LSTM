# 🤟 Sign Language Recognition using LSTM & Locking System

> Real-time hand gesture recognition system achieving **99%+ accuracy**, built to bridge communication barriers for the deaf and mute community.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Detection-green?logo=google)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25+-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project presents **SLR-LSTM-LS** (Sign Language Recognition using Long Short-Term Memory and Locking System) — an end-to-end Android application that translates hand gestures into text and speech in real-time.

Unlike prior work relying on CNN-based systems (68–97% accuracy), our **novel locking mechanism** filters low-confidence predictions and locks only gestures exceeding 90% confidence — resulting in a mean accuracy of **99%+** across all tested signs.

**Recognised Signs:**
| Sign | Accuracy |
|------|----------|
| Nice to meet you | 99.90% |
| How are you | 99.93% |
| Hello | 99.90% |
| Thank you | 99.80% |
| What is your name | 99.00% |

---

## 🏗️ System Architecture

```
Webcam Input
    │
    ▼
Hand Detection (MediaPipe landmarks)
    │
    ▼
Pre-processing (background removal, resize, coordinate extraction)
    │
    ▼
Feature Extraction → .npy format
    │
    ▼
LSTM Model (Sequential, trained on custom dataset)
    │
    ▼
Lock System (confidence > 90% threshold)
    │
    ▼
Text Output + Text-to-Speech
```

---

## ✨ Key Features

- **Custom Dataset** — 30 images per gesture, 5 gesture classes, built from scratch using webcam
- **LSTM-based Recognition** — Leverages sequential modelling for temporal gesture patterns
- **Locking System** — Novel accuracy filter: only accepts predictions above 90% confidence, eliminating false positives
- **Real-time Processing** — Runs on standard hardware (tested on Intel Core i7, 8GB RAM)
- **Text-to-Speech Output** — Converts recognised gestures to audible speech
- **Android Integration** — Deployed as a mobile application via Android Studio

---

## 🛠️ Tech Stack

| Area | Technology |
|------|-----------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Hand Detection | MediaPipe |
| IDE | PyCharm, Android Studio |
| Model Format | `.h5`, `.h` (Keras) |
| Data Format | NumPy `.npy` |

---


## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

---

### Step 1 — Collect Training Data

```bash
python collectdata.py
```

Opens your webcam and captures **30 images per gesture** into the `Images/` folder.
Repeat for each gesture class you want to recognise.

---

### Step 2 — Extract Features

```bash
python data.py
```

Processes raw images through **MediaPipe Hand Landmark Detection** and saves
extracted landmark arrays as `.npy` files into the `MP_Data/` folder.

---

### Step 3 — Train the LSTM Model

```bash
python trainmodel.py
```

Trains the sequential LSTM model on the extracted features.
Saves the trained model as `model.h5` and `model.json`.
Training logs are written to `Logs/train/` — view with TensorBoard:

```bash
tensorboard --logdir Logs/train
```

---

### Step 4 — Run Real-Time Recognition

```bash
python main.py
```

Opens your webcam, detects hand gestures via MediaPipe, and applies the
**locking system** (confidence > 90%) for high-accuracy recognition.
Recognised gestures are converted to speech via the TTS module.

## 📊 Results

Our system outperforms previous vision-based hand gesture recognition approaches:

| System | Accuracy Range | Mean Accuracy |
|--------|---------------|---------------|
| Vision-based (Kinect Sensor) [Al Farid et al., 2022] | 68% – 97% | 86.6% |
| CNN Fine-tuned (AlexNet/VGG16) [Sahoo et al., 2022] | 64% – 98.14% | ~85% |
| **SLR-LSTM-LS (Ours)** | **99% – 99.93%** | **99.7%** |

---

## 🔮 Future Work

- Expand gesture vocabulary beyond 5 signs
- Improve robustness under varying lighting conditions
- Integrate with a real-time video call application
- Add multilingual sign language support (e.g., BSL, ASL, ISL)

---

## 👥 Authors

- **Saddam Ahmed Nisar** (192101024)
- **Muhammad Touseef** (192101002)

Supervised by: Dr. Usman Ali Gulzari — KICSIT Sub Campus of IST

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- Al Farid et al. (2022). A Structured and Methodological Review on Vision-Based Hand Gesture Recognition System. *Journal of Imaging*, 8(6), 153.
- Sahoo et al. (2022). Real-Time Hand Gesture Recognition Using Fine-Tuned Convolutional Neural Network. *Sensors*, 22(3), 706.
- [MediaPipe Gesture Recognition](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)
