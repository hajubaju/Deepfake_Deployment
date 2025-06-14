# 🔊 Deepfake Audio Detection (Raspberry Pi 4)

This project implements a lightweight, offline-ready deepfake audio detection system on a Raspberry Pi 4 (4GB). It allows users to upload audio files from a browser (e.g., a phone) and receive a prediction — either `REAL` or `FAKE` — along with confidence.

The backend uses a pre-trained TensorFlow Lite model to run inference on-device. The frontend is a simple web UI built with Flask.

---

## 🧠 Features

- ✅ Audio file upload from any phone or PC (same network as Raspberry Pi)
- ✅ Supports `.flac`, `.wav`, `.mp3`, and `.m4a`
- ✅ Works offline — perfect for secure environments
- ✅ Real-time results with confidence score
- ✅ Custom audio feature extraction (wavelet + spectral + pitch-based)
- ✅ Fast inference using TensorFlow Lite

---

## 🚀 Requirements

### 🔧 Raspberry Pi Setup

- Raspberry Pi 4 (4GB recommended)
- Python 3.7+
- Internet connection (initial setup only)

### 📦 Dependencies

Install these once on your Pi:

```bash
sudo apt update
sudo apt install python3-pip libsndfile1 libatlas-base-dev -y

pip install flask werkzeug numpy scipy librosa soundfile pandas pywt joblib
pip install tensorflow  # or tflite-runtime if you're optimizing for size

🌐 Running Locally (on your Pi)
Clone or copy this project to your Raspberry Pi.

Start the Flask server:
Install the requirements 
python3 app.py