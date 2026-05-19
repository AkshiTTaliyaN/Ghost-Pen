# GhostPen — Real-Time Air Drawing & Gesture Recognition

> No touch. No stylus. Just your hand and a webcam.

![Python](https://img.shields.io/badge/Python-3.9+-00ff9f?style=flat-square&logo=python&logoColor=black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-00cfff?style=flat-square&logo=tensorflow&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-00ff9f?style=flat-square&logo=opencv&logoColor=black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00cfff?style=flat-square&logo=google&logoColor=black)

---

## ✍️ What It Does

GhostPen tracks your index fingertip in real time via webcam and recognises letters and symbols you draw in the air, powered by a custom-trained CNN:

| Module | What It Does |
|---|---|
| 🖐️ **Hand Tracker** | MediaPipe detects 21 hand landmarks per frame at 30+ FPS |
| 🎯 **Stroke Capture** | Fingertip coordinates are plotted onto a virtual canvas in real time |
| 🧠 **CNN Classifier** | Custom-trained model predicts the drawn character with **90% accuracy** |
| 🖥️ **Live Display** | OpenCV window renders the canvas, landmark overlay, and predicted label |

Every stroke is captured, preprocessed, and classified in a single pipeline with no perceptible lag.

---

## 🗂️ Project Structure

```
ghostpen/
├── main.py                 # Entry point, launches webcam and pipeline
├── requirements.txt        # Python dependencies
├── README.md
└── modules/
    ├── tracker.py          # MediaPipe hand landmark detection
    ├── canvas.py           # Stroke capture and virtual drawing canvas
    ├── preprocessor.py     # Frame normalization before model input
    ├── classifier.py       # CNN model loader and inference
    └── dataset/            # Training data and label mappings
```

---

## 🧠 How It Works

```
Webcam Frame → MediaPipe Hand Tracking → Fingertip Coordinates
     → Stroke Canvas → Preprocessor → CNN Model → Predicted Character
```

Each frame goes through the full pipeline. The model only classifies when a complete stroke is detected reducing false positives mid-draw.

---

## 🧮 Model Performance

| Metric | Value |
|---|---|
| Overall Accuracy | **90%** |
| Input | Grayscale 28×28 stroke image |
| Architecture | Custom CNN (Conv → Pool → Dense) |
| Framework | TensorFlow / Keras |
| Training Data | Custom gesture dataset (ongoing expansion) |

---

## 🛡️ Design Decisions

- **Single responsibility** — tracking, capture, preprocessing, and inference are fully decoupled modules
- **No external API calls** — fully offline, runs on local hardware only
- **Stroke gating** — model inference only triggers on completed strokes, not every frame
- **Graceful degradation** — if no hand is detected, canvas holds last state without crashing
- **Custom dataset** — trained on self-collected gesture data, not a generic public dataset

---

## 🚀 Getting Started

```bash
git clone https://github.com/AkshiTTaliyaN/Ghost-Pen
cd Ghost-Pen
pip install -r requirements.txt
python main.py
```

> Make sure your webcam is connected and well-lit. Hold your index finger up and draw, GhostPen will do the rest.

---

## 📍 Current Status

- [x] Real-time hand tracking via MediaPipe
- [x] Custom CNN trained and deployed
- [x] 90% classification accuracy
- [ ] Expanding dataset with more characters and symbols
- [ ] Streamlit web interface
- [ ] Multi-hand and multi-stroke support
- [ ] Export drawn text as string output

---

## ⚠️ Usage Note

GhostPen is a research and portfolio project.  
Performance may vary based on lighting conditions, webcam quality, and background contrast.  
