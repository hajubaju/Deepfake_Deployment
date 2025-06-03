import numpy as np
import pywt
import librosa
import pandas as pd
import logging
import joblib
import soundfile as sf
from pywt import WaveletPacket
from scipy.stats import entropy
import tflite_runtime.interpreter as tflite
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load pre-fitted preprocessing artifacts
scaler = joblib.load('robust_scaler1.pkl')
transformer = joblib.load('yeo_johnson_transform.pkl')

# Load and normalize audio
def load_audio(filename):
    try:
        y, sr = librosa.load(filename, sr=16000, mono=True)  # handles .flac, .wav, .mp3 etc.
        logging.info(f"Loaded using librosa: {filename}")
    except Exception as e:
        logging.warning(f"librosa failed: {e}. Trying soundfile.")
        try:
            y, sr = sf.read(filename)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            logging.info(f"Loaded using soundfile: {filename}")
        except Exception as e2:
            logging.error(f"soundfile also failed: {e2}")
            return None, None
    if y is None or len(y) == 0 or np.max(np.abs(y)) == 0:
        logging.error("Audio is empty or silent")
        return None, None
    y = y / np.max(np.abs(y))
    return sr, y

# Decomposition functions
def ewt_decompose(signal):
    return pywt.wavedec(signal, 'db4', level=4)

def wpt_decompose(signal, wavelet='db4', level=3):
    wp = WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    return [wp[node.path].data for node in wp.get_level(level)]

# Feature extraction
def extract_wavelet_features(ewt, wpt):
    features = []
    for coeff in ewt + wpt:
        coeff = np.abs(coeff) + 1e-10
        features.extend([np.mean(coeff), np.var(coeff), entropy(coeff)])
    return features

def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return float(1 - np.std(mfcc) / np.mean(mfcc))

def extract_pitch_variation(y, sr):
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    valid = pitches[mags > np.median(mags)]
    return float(np.std(valid) / np.mean(valid)) if len(valid) > 0 and np.mean(valid) > 0 else 0.0

def extract_features(file_path):
    try:
        sr, y = load_audio(file_path)
        if y is None:
            raise ValueError("Invalid or empty audio")
        ewt = ewt_decompose(y)
        wpt = wpt_decompose(y)
        feats = extract_wavelet_features(ewt, wpt)
        spec_feat = extract_spectral_feature(y, sr)
        pitch_feat = extract_pitch_variation(y, sr)
        full_vector = [spec_feat, pitch_feat] + feats
        return np.array(full_vector, dtype=np.float32).reshape(1, -1)
    except Exception:
        logging.exception(f"❌ Feature extraction failed for {file_path}")
        return None

# Inference
def run_inference(file_path):
    try:
        interpreter = tflite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        features = extract_features(file_path)
        if features is None:
            return {"status": "error", "message": "Feature extraction failed"}

        features = transformer.transform(features)
        if hasattr(scaler, 'feature_names_in_'):
            df = pd.DataFrame(features, columns=scaler.feature_names_in_)
            features_scaled = scaler.transform(df)
        else:
            features_scaled = scaler.transform(features)

        features_scaled = features_scaled.astype(np.float32)
        if features_scaled.shape[1] != input_details[0]['shape'][1]:
            return {"status": "error", "message": "Input shape mismatch with model"}

        interpreter.set_tensor(input_details[0]['index'], features_scaled)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        prob = float(output[0][0])
        pred = int(prob > 0.5)

        return {
            "file": file_path,
            "status": "success",
            "label": "Bona fide" if pred else "Spoof",
            "prediction": pred,
            "confidence": round(prob if pred else 1 - prob, 4)
        }
    except Exception as e:
        logging.exception("❌ Inference failed")
        return {"status": "error", "message": f"Inference error: {str(e)}"}
