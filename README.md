
# 🏃 Personalized Running Posture Feedback System using MediaPipe + Generative AI

This project provides an intelligent and accessible system for analyzing running posture from a single video clip, offering personalized feedback for injury prevention and performance improvement. It combines MediaPipe pose estimation, biomechanical feature extraction, deep learning (Autoencoder), and GPT-generated natural language feedback into a streamlined platform.

## Project Overview

Running is one of the most common forms of physical activity, but many amateur runners unknowingly adopt poor movement patterns that increase the risk of injury. Traditionally, analyzing a runner’s form requires expensive hardware and expert supervision. This project addresses these challenges by offering an AI-powered video-based tool that can:

- Extract body landmarks using MediaPipe
- Calculate biomechanical metrics like joint angles, trunk lean, stride width, and arm swing
- Detect and segment gait cycles (Initial Contact, Mid Stance, Swing Phase, Toe Off)
- Normalize and compare motion with elite runners using Cosine Similarity, Statistic Deviation, Dynamic Time Warping (DTW), and a trained Autoencoder
- Generate phase-specific natural language feedback using OpenAI’s GPT tailored to each runner's movement profile

This tool helps runners self-assess their form and receive actionable insights using just a single video and modern AI techniques.

---

## Core Technologies

- **MediaPipe** – Real-time pose estimation
- **PyTorch** – Autoencoder model for anomaly detection
- **OpenAI GPT API** – For generating personalized feedback
- **Streamlit** – User-friendly web interface
- **Dynamic Time Warping (DTW), Statistic Deviation, and Cosine Similarity** – Biomechanical comparison
- **Pandas / NumPy / SciPy / Matplotlib** – Data processing and visualization

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prattapong/Personalized-Running-Posture-Analysis-feedback-for-injuries-prevention-using-MediaPipe.git
cd Personalized-Running-Posture-Analysis-feedback-for-injuries-prevention-using-MediaPipe
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your .env file

```
OPENAI_API_KEY=sk-xxxxx
```

---

## How to Run the System

### Step 1: Train the Autoencoder (if not already trained)

Run the following command to process elite runner videos and train the model:

```bash
python autoencoder_best_model.py
```

This script will:
- Process videos to extract pose features
- Segment gait cycles
- Normalize motion patterns
- Train a deep autoencoder
- Save the trained model as best_model.pth

> You can modify this script to use your own videos or elite datasets.

### Step 2: Upload best_model.pth manually

Before running the app, you must upload or move your trained model file (best_model.pth) to the following directory:

```swift
C:/Users/User/OneDrive/Desktop/Deepproject/MODEL/best_model.pth
```
⚠️ If this file is missing, the app will not work.

### Step 3: Launch the Streamlit Web App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.
The app allows users to:
- Upload running videos
- Analyze gait features in real time
- Compare their motion with elite runners
- View GPT-generated natural language feedback organized by gait phase

---

## 🧑‍💻 Contributors

**Graduate School of Applied Statistics, National Institute of Development Administration (NIDA)**
- Rattapong Pojpatinya — 6620422008@stu.nida.ac.th
- Tuksaporn Chaiyaraks — 6620422013@stu.nida.ac.th
- Pakawat Naktubtee — 6620422021@stu.nida.ac.th
- Kanpitcha Panbualuang — 6620422024@stu.nida.ac.th
- Kittitat Wattanasuntikul — 6620422025@stu.nida.ac.th
- **Advisor**: Dr. Thitirat Siriborvornratanakul — thitirat@as.nida.ac.th

---

## 📜 License

This project is for educational and research purposes. Contact the authors for permission if you'd like to use this in production.

---

## 🔑 Keywords
Pose estimation, Biomechanics, Gait analysis, Running injury prevention, MediaPipe, Autoencoder, Generative AI, DTW, GPT-4, Running form feedback
