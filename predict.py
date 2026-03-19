import numpy as np
import cv2
from scipy import signal
from scipy.stats import skew, kurtosis
import os
import sys
import json
import joblib

# Add parent directory to path to import ml_models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models import predict_with_tabular_model_vector, get_metadata, LABELS

labels = ['ht1', 'ht2', 'nt', 'pt']

def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img


def extract_ppg_features_from_image(image_path):
    """
    Extract PPG signal features from a PPG waveform image.
    The image is assumed to show PPG signal over time.
    Returns a feature vector compatible with trained models.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Crop to the signal area (remove margins, focus on the waveform)
        height, width = gray.shape
        signal_start_row = int(height * 0.1)
        signal_end_row = int(height * 0.9)
        signal_region = gray[signal_start_row:signal_end_row, :]
        
        # Invert if needed (darker signal on lighter background)
        if np.mean(signal_region) < 128:
            signal_region = 255 - signal_region
        
        # Extract vertical profile (PPG signal intensity across time/columns)
        ppg_signal = np.mean(signal_region, axis=0)
        
        # Normalize signal to 0-1 range
        if len(ppg_signal) > 0:
            ppg_signal = (ppg_signal - ppg_signal.min()) / (ppg_signal.max() - ppg_signal.min() + 1e-8)
        
        # Extract features from the signal
        features = {}
        
        # 1. Basic statistics
        features['mean'] = float(np.mean(ppg_signal))
        features['std'] = float(np.std(ppg_signal))
        features['min'] = float(np.min(ppg_signal))
        features['max'] = float(np.max(ppg_signal))
        features['range'] = float(np.max(ppg_signal) - np.min(ppg_signal))
        features['median'] = float(np.median(ppg_signal))
        
        # 2. Energy and variance
        features['variance'] = float(np.var(ppg_signal))
        features['energy'] = float(np.sum(ppg_signal ** 2))
        features['rms'] = float(np.sqrt(np.mean(ppg_signal ** 2)))
        
        # 3. First derivative (rate of change)
        if len(ppg_signal) > 1:
            first_deriv = np.diff(ppg_signal)
            features['first_deriv_mean'] = float(np.mean(np.abs(first_deriv)))
            features['first_deriv_std'] = float(np.std(first_deriv))
            features['first_deriv_max'] = float(np.max(np.abs(first_deriv)))
        else:
            features['first_deriv_mean'] = 0.0
            features['first_deriv_std'] = 0.0
            features['first_deriv_max'] = 0.0
        
        # 4. Second derivative (acceleration)
        if len(ppg_signal) > 2:
            second_deriv = np.diff(ppg_signal, n=2)
            features['second_deriv_mean'] = float(np.mean(np.abs(second_deriv)))
            features['second_deriv_std'] = float(np.std(second_deriv))
        else:
            features['second_deriv_mean'] = 0.0
            features['second_deriv_std'] = 0.0
        
        # 5. Peak detection (systolic/diastolic peaks in PPG)
        try:
            peaks, _ = signal.find_peaks(ppg_signal, height=np.max(ppg_signal) * 0.3, distance=10)
            features['peak_count'] = float(len(peaks))
            if len(peaks) > 0:
                peak_heights = ppg_signal[peaks]
                features['peak_mean_height'] = float(np.mean(peak_heights))
                features['peak_max_height'] = float(np.max(peak_heights))
                features['peak_std_height'] = float(np.std(peak_heights))
            else:
                features['peak_mean_height'] = 0.0
                features['peak_max_height'] = 0.0
                features['peak_std_height'] = 0.0
        except:
            features['peak_count'] = 0.0
            features['peak_mean_height'] = 0.0
            features['peak_max_height'] = 0.0
            features['peak_std_height'] = 0.0
        
        # 6. Valley detection (diastolic valleys)
        try:
            valleys, _ = signal.find_peaks(-ppg_signal, height=-np.min(ppg_signal) * 0.7, distance=10)
            features['valley_count'] = float(len(valleys))
            if len(valleys) > 0:
                valley_heights = ppg_signal[valleys]
                features['valley_mean_height'] = float(np.mean(valley_heights))
                features['valley_min_height'] = float(np.min(valley_heights))
            else:
                features['valley_mean_height'] = 0.0
                features['valley_min_height'] = 0.0
        except:
            features['valley_count'] = 0.0
            features['valley_mean_height'] = 0.0
            features['valley_min_height'] = 0.0
        
        # 7. Amplitude and pulse-related features
        features['amplitude'] = float(np.max(ppg_signal) - np.min(ppg_signal))
        if features['peak_count'] > 0:
            features['systolic_diastolic_ratio'] = float((np.max(ppg_signal) - np.mean(ppg_signal)) / (np.mean(ppg_signal) - np.min(ppg_signal) + 1e-8))
        else:
            features['systolic_diastolic_ratio'] = 1.0
        
        # 8. Distribution shape
        try:
            features['skewness'] = float(skew(ppg_signal))
            features['kurtosis'] = float(kurtosis(ppg_signal))
        except:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        
        # 9. Frequency domain features (using FFT)
        try:
            fft_vals = np.abs(np.fft.fft(ppg_signal))
            fft_freqs = np.fft.fftfreq(len(ppg_signal))
            features['fft_max_magnitude'] = float(np.max(fft_vals))
            features['fft_mean_magnitude'] = float(np.mean(fft_vals))
            features['fft_std_magnitude'] = float(np.std(fft_vals))
        except:
            features['fft_max_magnitude'] = 0.0
            features['fft_mean_magnitude'] = 0.0
            features['fft_std_magnitude'] = 0.0
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from image: {e}")
        return None


def get_expected_feature_names():
    """Get the expected feature names from trained model metadata."""
    try:
        meta = get_metadata()
        if meta and 'feature_names' in meta:
            return meta['feature_names']
    except:
        pass
    return None


def align_features_to_model(extracted_features):
    """
    Align extracted features to match the trained model's expected feature order.
    Returns a feature vector in the correct order.
    """
    expected_names = get_expected_feature_names()
    
    if expected_names is None:
        # If no metadata, use all extracted features as vector
        return list(extracted_features.values()) if isinstance(extracted_features, dict) else extracted_features
    
    # Create vector in expected order
    feature_vector = []
    for name in expected_names:
        if name in extracted_features:
            feature_vector.append(float(extracted_features[name]))
        else:
            feature_vector.append(0.0)
    
    return feature_vector


# --- CNN helper: load & predict -------------------------------------------
_CNN_MODEL_CACHE = {}

def _load_cnn_model(model_path=None):
    """Load Keras/TensorFlow model from models/cnn.h5 and cache it."""
    try:
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'cnn.h5')
        if model_path in _CNN_MODEL_CACHE:
            return _CNN_MODEL_CACHE[model_path]
        # import locally to avoid hard dependency unless used
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        _CNN_MODEL_CACHE[model_path] = model
        return model
    except Exception as e:
        print(f"Could not load CNN model from {model_path}: {e}")
        return None


def _predict_with_cnn(image_path, model_path=None):
    """Run the CNN model on the image and return (label, confidence)."""
    try:
        model = _load_cnn_model(model_path)
        if model is None:
            return None, 0.0

        # Preprocess image for common CNNs (224x224 RGB, scaled)
        img = cv2.imread(image_path)
        if img is None:
            return None, 0.0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)).astype('float32') / 255.0
        x = np.expand_dims(img_resized, axis=0)

        preds = model.predict(x)
        # preds may be logits or probabilities
        preds = np.array(preds).squeeze()
        if preds.ndim == 0:
            preds = np.array([preds])
        # Softmax if not normalized
        if preds.sum() <= 0 or not np.isclose(preds.sum(), 1.0):
            try:
                ex = np.exp(preds - np.max(preds))
                probs = ex / (np.sum(ex) + 1e-8)
            except Exception:
                probs = np.ones_like(preds) / len(preds)
        else:
            probs = preds

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        # Map index -> label using stored label encoder if available
        label = None
        try:
            # try to load saved label encoder from models
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            le_path = os.path.join(models_dir, 'label_encoder.joblib')
            if os.path.isfile(le_path):
                le = joblib.load(le_path)
                label = le.inverse_transform([idx])[0]
        except Exception:
            label = None

        if label is None:
            # fallback to default LABELS ordering
            if 0 <= idx < len(LABELS):
                label = LABELS[idx]
            else:
                label = LABELS[0]

        return label, confidence
    except Exception as e:
        print(f"Error during CNN prediction: {e}")
        return None, 0.0


def run_prediction(path, model_name):
    """
    Analyze uploaded PPG signal image and predict hypertension class.
    For `cnn` uses a Keras model (models/cnn.h5). For tabular models uses extracted PPG features.
    """
    try:
        if model_name and model_name.lower() == 'cnn':
            lab, conf = _predict_with_cnn(path)
            if lab is None:
                return 'nt', 0.5
            return lab, float(conf)

        # For other models (knn, svm, xgboost): extract PPG features and use tabular predictors
        extracted_features = extract_ppg_features_from_image(path)
        if extracted_features is None:
            return 'nt', 0.5

        model_key = model_name
        # Align and predict with tabular model
        feature_vector = align_features_to_model(extracted_features)
        predicted_class, confidence = predict_with_tabular_model_vector(model_key, feature_vector)
        if predicted_class is None or predicted_class == '':
            return 'nt', 0.5
        return predicted_class, float(max(0.0, min(1.0, confidence)))

    except Exception as e:
        print(f"Error in run_prediction: {e}")
        import traceback
        traceback.print_exc()
        return 'nt', 0.5
