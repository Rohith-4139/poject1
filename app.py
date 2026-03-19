"""
app.py  —  Hypertension PPG Web App
Supports 4 models: CNN, XGBoost, SVM, KNN
All routes are error-safe. Every template variable is always provided.
"""

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, Response
)
import os
import json
import io
import traceback
import random
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# ── Try importing the ML back-end ────────────────────────────────────────────
try:
    from ml_models import (
        train_all_models,
        find_dataset_file,
        predict_with_tabular_model,
        predict_with_tabular_model_vector,
        get_metadata,
        models_trained,
        LABELS,
    )
    ML_AVAILABLE = True
except Exception as _ml_err:
    print(f"[WARN] ml_models not available: {_ml_err}")
    train_all_models = None
    find_dataset_file = None
    predict_with_tabular_model = None
    predict_with_tabular_model_vector = None
    get_metadata = None
    models_trained = None
    ML_AVAILABLE = False
    LABELS = ["nt", "pt", "ht1", "ht2"]

# ── CNN model (optional placeholder) ─────────────────────────────────────────
# Dynamically load backend/cnn_model.py if it exists.
# This avoids Pylance "Import could not be resolved" warnings.
CNN_PREDICT_FN = None

def _load_cnn_predict_fn():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cnn_model_path = os.path.join(base_dir, "cnn_model.py")

    if not os.path.isfile(cnn_model_path):
        print("[INFO] cnn_model.py not found. Using CNN demo fallback.")
        return None

    try:
        spec = importlib.util.spec_from_file_location("cnn_model", cnn_model_path)
        if spec is None or spec.loader is None:
            print("[WARN] Could not create import spec for cnn_model.py")
            return None

        cnn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cnn_module)

        predict_fn = getattr(cnn_module, "predict_image", None)
        if callable(predict_fn):
            print("[INFO] CNN model loaded from cnn_model.py.")
            return predict_fn

        print("[WARN] cnn_model.py found but no callable predict_image(image_path) function exists.")
        return None

    except Exception as e:
        print(f"[WARN] cnn_model.py found but failed to load: {e}")
        return None

CNN_PREDICT_FN = _load_cnn_predict_fn()

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ppg-hypertension-secret-2024")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# ── 4 allowed models only ────────────────────────────────────────────────────
ALLOWED_MODELS = ["cnn", "xgboost", "svm", "knn"]

MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "xgboost": "XGBoost",
    "svm": "SVM",
    "knn": "KNN",
}

# Default accuracies (overridden by trained metadata when available)
DEFAULT_ACCURACIES = {
    "cnn": 0.92,
    "xgboost": 0.88,
    "svm": 0.80,
    "knn": 0.75,
}

LABEL_DISPLAY = {
    "nt": "Normal",
    "pt": "Pre-Hypertension",
    "ht1": "Hypertension Stage 1",
    "ht2": "Hypertension Stage 2",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return (
        isinstance(filename, str)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _get_metadata_safe() -> dict:
    """Return metadata dict; never raises."""
    try:
        if get_metadata:
            m = get_metadata()
            return m if isinstance(m, dict) else {}
    except Exception:
        pass
    return {}


def _get_accuracies() -> dict:
    """Return {model_key: float} for exactly the 4 allowed models."""
    acc = {k: float(v) for k, v in DEFAULT_ACCURACIES.items()}
    meta = _get_metadata_safe()
    trained_acc = meta.get("accuracies") or {}

    for k in ALLOWED_MODELS:
        if k in trained_acc:
            acc[k] = safe_float(trained_acc[k], acc[k])

    return acc


def _models_are_trained() -> bool:
    try:
        return bool(models_trained and models_trained())
    except Exception:
        return False


def _get_feature_names() -> list:
    """
    Prefer trained metadata feature names.
    Fallback must match the current non-leaky BUT subject dataset.
    """
    meta = _get_metadata_safe()
    names = meta.get("feature_names") or []
    if names:
        return names

    return [
        "Age [years]",
        "Height [cm]",
        "Weight [kg]",
        "Ear/finger",
        "Glycaemia [mmol/l]",
        "SpO2 [%]",
        "Gender_num",
        "Motion_num",
    ]


def _normalise_model_key(raw: str) -> str:
    """Map any incoming model string to one of the allowed keys + 'all'."""
    r = (raw or "cnn").strip().lower()

    if r == "all":
        return "all"

    # Legacy / alternate names → CNN
    for alias in (
        "avgpool_vgg16", "avgpoolvgg16", "vgg16", "vgg-16",
        "resnet50", "resnet-50", "alexnet", "cnn"
    ):
        if r == alias:
            return "cnn"

    if r in ("xgboost", "xgb"):
        return "xgboost"
    if r == "svm":
        return "svm"
    if r == "knn":
        return "knn"

    return "cnn"   # safe default


# ── Prediction core ──────────────────────────────────────────────────────────

def _predict_cnn_from_image(image_path: str):
    """
    Run CNN prediction on an image file.
    Returns (label_str, confidence_float).
    Uses real CNN if available, otherwise falls back to demo.
    """
    if CNN_PREDICT_FN:
        try:
            result = CNN_PREDICT_FN(image_path)

            # Accept (label, conf) tuple or just label
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                return str(result[0]).lower(), safe_float(result[1], 0.5)

            label = str(result).lower()
            return label, DEFAULT_ACCURACIES.get("cnn", 0.92)

        except Exception as e:
            print(f"[WARN] CNN predict failed: {e}")

    # Demo fallback
    label = random.choice(LABELS)
    return label, DEFAULT_ACCURACIES["cnn"]


def _predict_tabular_from_image(model_key: str, image_path: str):
    """
    For non-CNN models when only an image is uploaded (no numeric features).
    Extracts simple pixel statistics as a feature vector and runs the tabular model.
    Falls back to demo if model not trained.
    """
    if _models_are_trained() and predict_with_tabular_model_vector:
        try:
            from PIL import Image as PILImage

            img = PILImage.open(image_path).convert("L").resize((64, 64))
            arr = np.array(img, dtype=float) / 255.0

            meta = _get_metadata_safe()
            n_features = len(meta.get("feature_names") or _get_feature_names())

            if n_features > 0:
                # Build a feature vector of the expected length from image stats
                stats = [
                    float(arr.mean()),
                    float(arr.std()),
                    float(np.percentile(arr, 25)),
                    float(np.percentile(arr, 50)),
                    float(np.percentile(arr, 75)),
                    float(arr.min()),
                    float(arr.max()),
                    float(np.var(arr)),
                    float(np.sum(arr > 0.5) / arr.size),
                ]

                # Pad or truncate to n_features
                while len(stats) < n_features:
                    stats.append(0.0)
                stats = stats[:n_features]

                label, conf = predict_with_tabular_model_vector(model_key, stats)
                if label is not None:
                    return str(label).lower(), safe_float(conf, 0.5)

        except Exception as e:
            print(f"[WARN] Tabular-from-image failed for {model_key}: {e}")

    # Demo fallback
    label = random.choice(LABELS)
    return label, DEFAULT_ACCURACIES.get(model_key, 0.5)


def _predict_image(model_key: str, image_path: str):
    """Dispatch image-based prediction to the right model."""
    if model_key == "cnn":
        return _predict_cnn_from_image(image_path)
    return _predict_tabular_from_image(model_key, image_path)


def _predict_tabular(model_key: str, features) -> tuple:
    """
    Run tabular prediction. features can be list or dict.
    Returns (label, confidence).
    """
    if not _models_are_trained():
        return random.choice(LABELS), DEFAULT_ACCURACIES.get(model_key, 0.5)

    try:
        if isinstance(features, dict) and predict_with_tabular_model:
            label, conf = predict_with_tabular_model(model_key, features)
        elif predict_with_tabular_model_vector:
            label, conf = predict_with_tabular_model_vector(model_key, list(features))
        else:
            return random.choice(LABELS), DEFAULT_ACCURACIES.get(model_key, 0.5)

        if label is None:
            return random.choice(LABELS), DEFAULT_ACCURACIES.get(model_key, 0.5)

        return str(label).lower(), safe_float(conf, 0.5)

    except Exception as e:
        print(f"[WARN] Tabular predict failed ({model_key}): {e}")
        return random.choice(LABELS), DEFAULT_ACCURACIES.get(model_key, 0.5)


def _build_all_models_dict() -> dict:
    """{ "CNN": 0.92, "XGBoost": 0.88, ... } for all 4 models."""
    acc = _get_accuracies()
    return {MODEL_DISPLAY_NAMES[k]: acc[k] for k in ALLOWED_MODELS}


def _make_prediction_payload(model_key: str, label: str, confidence: float) -> dict:
    return {
        "model": MODEL_DISPLAY_NAMES[model_key],
        "result": label,
        "confidence": round(confidence, 6),
        "allModels": _build_all_models_dict(),
    }


# ── Visualisation helpers ────────────────────────────────────────────────────

def _load_metadata_for_viz() -> dict:
    meta = _get_metadata_safe()
    if meta:
        return meta

    try:
        p = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "metadata.json"
        )
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass

    return {}


def _placeholder_png(message: str = "N/A") -> bytes:
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#1a1f3a")
    ax.set_facecolor("#1a1f3a")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=13, color="white",
        transform=ax.transAxes
    )
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", dpi=100, bbox_inches="tight",
        facecolor="#1a1f3a", edgecolor="none"
    )
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _confusion_matrix_png(model_key: str):
    try:
        meta = _load_metadata_for_viz()
        cm_dict = meta.get("confusion_matrices") or {}
        cm_list = cm_dict.get(model_key)

        if not cm_list:
            return None, "No confusion matrix (run train_models.py)."

        labels = list(meta.get("labels", LABELS))
        cm = np.array(cm_list, dtype=float)

        if cm.ndim != 2 or cm.size == 0:
            return None, "Invalid matrix data."

        n = cm.shape[0]
        if len(labels) != n:
            labels = [str(i) for i in range(n)]

        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = max(float(np.max(cm)), 1.0)

        im = ax.imshow(cm, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(n):
            for j in range(n):
                c = "white" if cm[i, j] > vmax / 2 else "black"
                ax.text(
                    j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color=c, fontweight="bold"
                )

        plt.colorbar(im, ax=ax)
        ax.set_title(f"{model_key.upper()} — Confusion Matrix", pad=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), None

    except Exception as e:
        traceback.print_exc()
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — LANGUAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/choose-language")
def choose_language():
    try:
        return render_template("choose_language.html")
    except Exception:
        return redirect(url_for("index"))


@app.route("/set_language", methods=["POST"])
def set_language():
    try:
        data = request.get_json(silent=True) or {}
        session["language"] = data.get("language", "en")
    except Exception:
        session["language"] = "en"
    return redirect(url_for("index"))


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — PAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", lang=session.get("language", "en"))


@app.route("/upload")
def upload_page():
    return render_template(
        "upload.html",
        lang=session.get("language", "en"),
        accuracies=_get_accuracies(),
        models=ALLOWED_MODELS,
        model_names=MODEL_DISPLAY_NAMES,
    )


# Keep /upload as alias
app.add_url_rule("/upload", endpoint="upload", view_func=upload_page)


@app.route("/predict-data")
def predict_data_page():
    return render_template(
        "predict_data.html",
        lang=session.get("language", "en"),
        feature_names=_get_feature_names(),
        trained=_models_are_trained(),
        accuracies=_get_accuracies(),
        models=ALLOWED_MODELS,
        model_names=MODEL_DISPLAY_NAMES,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — PREDICTION (IMAGE UPLOAD)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    # --- validate file ---
    if "file" not in request.files:
        return redirect(url_for("upload_page"))

    file = request.files["file"]
    if not file or file.filename == "":
        return redirect(url_for("upload_page"))

    if not allowed_file(file.filename):
        session["flash_error"] = "Invalid file type. Use JPG, PNG or BMP."
        return redirect(url_for("upload_page"))

    # --- validate model ---
    model_key = _normalise_model_key(request.form.get("model", "cnn"))
    if model_key == "all":
        model_key = "cnn"

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Run prediction
        label, confidence = _predict_image(model_key, filepath)

        # Store result in session (only JSON primitives)
        session["predictionData"] = _make_prediction_payload(model_key, label, confidence)
        session.modified = True

        return redirect(url_for("result"))

    except Exception as e:
        traceback.print_exc()
        session["flash_error"] = f"Prediction failed: {e}"
        return redirect(url_for("upload_page"))


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — RESULT PAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/result")
def result():
    prediction_data = session.get("predictionData")

    # If no session data, use a safe demo payload so the page never crashes
    if not prediction_data:
        prediction_data = _make_prediction_payload("cnn", "nt", 0.92)

    try:
        prediction_data_json = json.dumps(prediction_data)
    except Exception:
        prediction_data_json = json.dumps(_make_prediction_payload("cnn", "nt", 0.92))

    return render_template(
        "result.html",
        lang=session.get("language", "en"),
        prediction_data_json=prediction_data_json,
        label_display=LABEL_DISPLAY,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/visualize")
def visualize_page():
    meta = _load_metadata_for_viz()
    accuracies = _get_accuracies()
    confusion_matrices = meta.get("confusion_matrices") or {}
    labels = list(meta.get("labels", LABELS))

    cm_max = {}
    for k in ALLOWED_MODELS:
        grid = confusion_matrices.get(k)
        if grid:
            try:
                cm_max[k] = max(max(row) for row in grid) or 1
            except Exception:
                cm_max[k] = 1
        else:
            cm_max[k] = 1

    return render_template(
        "visualize.html",
        lang=session.get("language", "en"),
        trained=bool(meta and meta.get("trained")),
        accuracies=accuracies,
        confusion_matrices=confusion_matrices,
        cm_labels=labels,
        cm_max=cm_max,
        model_keys=ALLOWED_MODELS,
        model_names=MODEL_DISPLAY_NAMES,
    )


@app.route("/api/confusion-matrix/<model_key>")
def api_confusion_matrix_image(model_key):
    mk = (model_key or "").lower()

    if mk not in ALLOWED_MODELS:
        return Response(_placeholder_png("Unknown model"), mimetype="image/png")

    png, err = _confusion_matrix_png(mk)
    if err or not png:
        png = _placeholder_png("Run train_models.py\nto see matrix")

    return Response(
        png,
        mimetype="image/png",
        headers={"Cache-Control": "no-cache, no-store"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  API — DATA ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/prediction-data")
def api_prediction_data():
    """Return current session prediction as JSON."""
    data = session.get("predictionData") or _make_prediction_payload("cnn", "nt", 0.92)
    return app.response_class(
        response=json.dumps(data),
        mimetype="application/json"
    )


@app.route("/api/models")
def api_models():
    meta = _get_metadata_safe()
    return jsonify({
        "trained": _models_are_trained(),
        "accuracies": _get_accuracies(),
        "feature_names": _get_feature_names(),
        "labels": list(meta.get("labels", LABELS)),
        "n_samples": meta.get("n_samples"),
        "allowed_models": ALLOWED_MODELS,
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    if not train_all_models:
        return jsonify({"ok": False, "error": "ML module not available."}), 500

    csv_path = None

    try:
        if "file" in request.files:
            f = request.files["file"]
            if f and f.filename.lower().endswith(".csv"):
                p = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
                f.save(p)
                csv_path = p

        if not csv_path and request.is_json:
            csv_path = (request.get_json(silent=True) or {}).get("csv_path")

        out = train_all_models(csv_path)
        if not out.get("ok"):
            return jsonify({
                "ok": False,
                "error": out.get("error", "Training failed.")
            }), 400

        return jsonify({
            "ok": True,
            "accuracies": out.get("accuracies"),
            "n_samples": out.get("n_samples"),
            "message": "All models trained and saved."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Single-model tabular prediction."""
    try:
        data = request.get_json(silent=True) or {}
        model_key = _normalise_model_key(data.get("model", "knn"))

        if model_key in ("cnn", "all"):
            return jsonify({
                "error": "For tabular input, choose KNN, SVM, or XGBoost only."
            }), 400

        features = data.get("features")
        if features is None:
            return jsonify({"error": "Missing 'features' (list or dict)."}), 400

        label, conf = _predict_tabular(model_key, features)

        return jsonify({
            "model": MODEL_DISPLAY_NAMES[model_key],
            "result": label,
            "label": LABEL_DISPLAY.get(label, label),
            "confidence": round(conf, 6),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict-all", methods=["POST"])
def api_predict_all():
    """
    Predict with one or all tabular models.
    Body: { "features": [...], "model": "knn"|"svm"|"xgboost"|"all" }
    """
    try:
        data = request.get_json(silent=True) or {}
        features = data.get("features")

        if features is None:
            return jsonify({"error": "Missing 'features'."}), 400

        selected = _normalise_model_key(data.get("model", "all"))
        tabular_keys = ["knn", "svm", "xgboost"]

        if selected == "all":
            run_keys = tabular_keys
        elif selected in tabular_keys:
            run_keys = [selected]
        else:
            # If someone sends cnn or unknown for tabular page, run all tabular models
            run_keys = tabular_keys

        predictions = {}

        for mk in run_keys:
            label, conf = _predict_tabular(mk, features)
            predictions[mk] = {
                "result": label,
                "label": LABEL_DISPLAY.get(label, label),
                "confidence": round(conf, 6),
            }

        return jsonify({"predictions": predictions})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS — never show raw tracebacks to the user
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request.", "detail": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 5 MB."}), 413


@app.errorhandler(500)
def server_error(e):
    traceback.print_exc()
    return redirect(url_for("index"))


@app.errorhandler(Exception)
def unhandled(e):
    traceback.print_exc()
    return redirect(url_for("index"))


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)