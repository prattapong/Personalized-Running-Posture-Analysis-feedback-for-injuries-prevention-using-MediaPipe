
# 🏃 Personalized Running Posture Feedback System using MediaPipe + Generative AI

This project provides an intelligent and accessible system for analyzing running posture from a single video clip, offering personalized feedback for injury prevention and performance improvement. It combines MediaPipe pose estimation, biomechanical feature extraction, deep learning (Autoencoder), and GPT-generated natural language feedback into a streamlined platform.

## Project Overview

Running is one of the most common forms of physical activity, but many amateur runners unknowingly adopt poor movement patterns that increase the risk of injury. Traditionally, analyzing a runner’s form requires expensive hardware and expert supervision. This project addresses these challenges by offering an AI-powered video-based tool that can:

- Extract 2D body landmarks using MediaPipe
- Calculate biomechanical metrics like joint angles, trunk lean, stride width, and arm swing
- Detect and segment gait cycles (Initial Contact, Mid Stance, Swing Phase, Toe Off)
- Normalize and compare motion with elite runners using Cosine Similarity, Dynamic Time Warping (DTW), and a trained Autoencoder
- Generate phase-specific natural language feedback using OpenAI’s GPT tailored to each runner's movement profile

This tool helps runners self-assess their form and receive actionable insights using just a single video and modern AI techniques.

---

## Core Technologies

- **MediaPipe** – Real-time pose estimation
- **PyTorch** – Autoencoder model for anomaly detection
- **OpenAI GPT API** – For generating personalized feedback
- **Streamlit** – User-friendly web interface
- **Dynamic Time Warping (DTW) and Cosine Similarity** – Biomechanical comparison
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

### 1. Step 1: Train the Autoencoder (if not already trained)

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

---

## 📊 Dashboard Features (Streamlit)

- **Live Predictions**: Delay probability shown for each flight.
- **Recent Flights Table**: Shows scheduled/actual times, delay status, etc.
- **Metrics Cards**: Accuracy, delay rate, precision, recall, average delay.
- **Charts**:
  - Top departure/arrival airports
  - Delay heatmap by hour
  - Airline performance comparison
  - Delay probability time series

---

## 🛠️ Feature Engineering

Each Kafka message (flight) is converted to features:

- `dep_airport`, `arr_airport`
- `dep_terminal`, `arr_terminal`
- `airline`, `route`
- `hour`, `day_of_week`
- Label: `1` if arrival delay > 0 else `0`

---

## ✅ Performance Metrics

Metrics are updated in real-time as labeled flight data becomes available:

- **Accuracy**: % of correct predictions
- **Precision**: % of predicted delays that were actually delayed
- **Recall**: % of actual delays that were correctly predicted
- **Delay Rate**: % of total flights delayed

---

## 📈 Sample API Message (Kafka Producer)

```json
{
  "key": "EK313",
  "value": {
    "flight_date": "2025-06-18",
    "flight_status": "active",
    "departure": {
      "airport": "Haneda Airport", "timezone": "Asia/Tokyo", "iata": "HND",
      "icao": "RJTT", "terminal": "3", "gate": "109","delay": 37,
      "scheduled": "2025-06-18T00:05:00+00:00",
      "estimated": "2025-06-18T00:05:00+00:00",
      "actual": "2025-06-18T00:42:00+00:00",
      "estimated_runway": "2025-06-18T00:42:00+00:00",
      "actual_runway": "2025-06-18T00:42:00+00:00"
    },
    "arrival": {
      "airport": "Dubai", "timezone": "Asia/Dubai", "iata": "DXB",
      "icao": "OMDB", "terminal": "3", "gate": null, "delay": 3, "baggage": "1",
      "scheduled": "2025-06-18T05:45:00+00:00",
      "estimated": "2025-06-18T05:48:00+00:00",
      "actual": null, "estimated_runway": null, "actual_runway": null
    },
    "airline": {
      "name": "Emirates", "iata": "EK", "icao": "UAE"
    },
    "flight": {
      "number": "313", "iata": "EK313", "icao": "UAE313", "codeshared": null
    },
    "aircraft": null,
    "live": null
  }
}
```

---

## 🔮 Future Improvements

- 🧪 Try advanced classifiers: `HoeffdingTree`, `SGDClassifier`, etc.
- 🧠 Optimizers: Experiment with `Adam`, `RMSProp`, etc.
- 🔧 Hyperparameter tuning with `SuccessiveHalvingClassifier`
- 🌤 Integrate external features like weather, holidays
- 📦 Store model state and historical predictions

---

## 🧑‍💻 Contributors

- Rattapong Pojpatinya  
- Tuksaporn Chaiyaraks  
- Pakawat Naktubtee  
- Kanpitcha Panbualuang  
- Kittitat Wattanasuntikul

---

## 📜 License

This project is for educational and research purposes. Contact the authors for permission if you'd like to use this in production.
