#Utility functions for audio processing and inference
import numpy as np
import pywt
import librosa
import pandas as pd
import logging
import joblib
import soundfile as sf
from pywt import WaveletPacket
from scipy.stats import entropy
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
tflite = tf.lite
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load pre-fitted preprocessing artifacts
scaler = joblib.load('robust_scaler1.pkl')
transformer = joblib.load('yeo_johnson_transform.pkl')

# Load and normalize audio
"""
def load_audio(filename):
    try:
        y, sr = librosa.load(filename, sr=16000)
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

    #y = y.astype(np.float32)
    y = y / np.max(np.abs(y))
    return sr, y
"""
def load_audio(filename):
    y, sr = librosa.load(filename, sr=16000)  # Use consistent sampling rate
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
    for coeff in ewt:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    for coeff in wpt:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    return features

def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return 1 - np.std(mfcc) / np.mean(mfcc)

def extract_pitch_variation(y, sr):
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    valid = pitches[mags > np.median(mags)]
    return np.std(valid) / np.mean(valid) if len(valid) > 0 and np.mean(valid) > 0 else 0

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
"""
import numpy as np
import pywt
import librosa
import pandas as pd
import logging
import joblib
import soundfile as sf
import tempfile
import os
from pywt import WaveletPacket
from scipy.stats import entropy
import tensorflow as tf
tflite = tf.lite
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load pre-fitted preprocessing artifacts
scaler = joblib.load('robust_scaler1.pkl')
transformer = joblib.load('yeo_johnson_transform.pkl')


# Convert .flac to temporary .wav file
def convert_flac_to_wav(flac_path):
    try:
        data, sr = sf.read(flac_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, data, sr)
        return temp_wav.name
    except Exception as e:
        logging.error(f"Conversion failed for {flac_path}: {e}")
        return None


# Load and normalize audio
def load_audio(filename):
    ext = os.path.splitext(filename)[1].lower()
    temp_wav_path = None

    if ext == ".flac":
        temp_wav_path = convert_flac_to_wav(filename)
        if temp_wav_path is None:
            return None, None
        filename = temp_wav_path

    try:
        y, sr = librosa.load(filename, sr=16000)
        #y = y.astype(np.float32)
        #if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    except Exception as e:
        logging.error(f"Audio loading failed for {filename}: {e}")
        return None, None
    finally:
        if temp_wav_path:
            os.remove(temp_wav_path)

    return sr, y


# Decomposition functions
def ewt_decompose(signal):
    return pywt.wavedec(signal, 'db4', level=4)

def wpt_decompose(signal, wavelet='db4', level=3):
    wp = WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    return [wp[node.path].data for node in wp.get_level(level)]

# Feature extraction
def extract_wavelet_features(ewt_coeffs, wpt_coeffs):
    features = []
    for coeff in ewt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    for coeff in wpt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    return features


def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return (1 - np.std(mfcc) / np.mean(mfcc))

def extract_pitch_variation(y, sr):
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    valid = pitches[mags > np.median(mags)]
    return (np.std(valid) / np.mean(valid)) if len(valid) > 0 and np.mean(valid) > 0 else 0

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
"""
