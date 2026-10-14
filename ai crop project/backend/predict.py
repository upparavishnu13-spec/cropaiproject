"""
PlantCare AI - Inference and Image Prediction Module
Performs image validation, preprocessing, MobileNetV2 inference,
confidence calibration checks, and recommendation enrichment.
"""

import os
import json
import logging
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Optional, Tuple

from recommendations import get_recommendation, parse_class_name

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlantCare-Predict")

# Constants
TARGET_IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB
LOW_CONFIDENCE_THRESHOLD = 0.60  # 60% confidence threshold

# Global cached model and class indices
_MODEL = None
_CLASS_INDICES: Optional[Dict[str, str]] = None
_MODEL_LOADED = False
_MODEL_LOAD_ERROR: Optional[str] = None


def allowed_file(filename: str) -> bool:
    """Checks if the uploaded file has a supported image extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def load_class_indices(classes_path: Optional[str] = None) -> Dict[str, str]:
    """
    Loads the class index-to-label mapping from JSON.
    Default path: backend/models/class_indices.json
    """
    global _CLASS_INDICES
    if _CLASS_INDICES is not None:
        return _CLASS_INDICES

    path = classes_path or os.path.join(
        os.path.dirname(__file__), "models", "class_indices.json"
    )

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                # Ensure keys are strings representing indices
                _CLASS_INDICES = {str(k): str(v) for k, v in raw_data.items()}
                logger.info(f"Loaded {len(_CLASS_INDICES)} classes from {path}")
                return _CLASS_INDICES
        except Exception as e:
            logger.error(f"Error reading class indices from {path}: {e}")

    # Fallback default classes
    logger.warning("Using fallback default class list.")
    _CLASS_INDICES = {
        "0": "Apple___Apple_scab",
        "1": "Apple___Black_rot",
        "2": "Apple___Cedar_apple_rust",
        "3": "Apple___healthy",
        "4": "Corn_(maize)___Common_rust_",
        "5": "Corn_(maize)___Northern_Leaf_Blight",
        "6": "Corn_(maize)___healthy",
        "7": "Grape___Black_rot",
        "8": "Grape___healthy",
        "9": "Pepper__bell___Bacterial_spot",
        "10": "Pepper__bell___healthy",
        "11": "Potato___Early_blight",
        "12": "Potato___Late_blight",
        "13": "Potato___healthy",
        "14": "Tomato___Early_blight",
        "15": "Tomato___Late_blight",
        "16": "Tomato___Leaf_Mold",
        "17": "Tomato___Septoria_leaf_spot",
        "18": "Tomato___Yellow_Leaf_Curl_Virus",
        "19": "Tomato___healthy"
    }
    return _CLASS_INDICES


def load_model(model_path: Optional[str] = None):
    """
    Loads the trained MobileNetV2 Keras model.
    If unavailable or TensorFlow is not installed, flags state gracefully.
    """
    global _MODEL, _MODEL_LOADED, _MODEL_LOAD_ERROR
    if _MODEL_LOADED and _MODEL is not None:
        return _MODEL

    path = model_path or os.getenv(
        "MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "models", "plant_disease_mobilenetv2.keras")
    )

    if not os.path.exists(path):
        # Also check for .h5 legacy format
        h5_path = path.replace(".keras", ".h5")
        if os.path.exists(h5_path):
            path = h5_path
        else:
            _MODEL_LOADED = False
            _MODEL_LOAD_ERROR = f"Model file not found at '{path}'. Run train_model.py to train."
            logger.info(_MODEL_LOAD_ERROR)
            return None

    try:
        import tensorflow as tf
        logger.info(f"Loading TensorFlow model from: {path}")
        _MODEL = tf.keras.models.load_model(path)
        _MODEL_LOADED = True
        _MODEL_LOAD_ERROR = None
        logger.info("Successfully loaded trained plant disease model.")
        return _MODEL
    except Exception as e:
        _MODEL_LOADED = False
        _MODEL_LOAD_ERROR = f"Failed to load model: {str(e)}"
        logger.error(_MODEL_LOAD_ERROR)
        return None


def get_model_status() -> Dict[str, Any]:
    """Returns the operational status of the ML model."""
    classes = load_class_indices()
    model = load_model()
    return {
        "model_loaded": _MODEL_LOADED and model is not None,
        "classes_count": len(classes),
        "status_detail": _MODEL_LOAD_ERROR if not _MODEL_LOADED else "Model operational",
        "supported_classes": list(classes.values())
    }


def validate_and_preprocess_image(file_stream) -> Tuple[np.ndarray, Image.Image]:
    """
    Validates file stream as a readable image and preprocesses for MobileNetV2:
    - Verifies file integrity
    - Converts to 3-channel RGB (removes alpha channels/greyscale)
    - Resizes to (224, 224) using high-quality Lanczos resampling
    - Applies MobileNetV2 normalization: scales pixels to [-1.0, 1.0]
    Returns (preprocessed_numpy_batch, original_pil_image).
    """
    try:
        # Load and verify integrity with PIL
        pil_img = Image.open(file_stream)
        pil_img.verify()

        # Re-open after verify() because verify() clears file pointer/state
        file_stream.seek(0)
        img = Image.open(file_stream)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"Corrupt or unreadable image file: {str(e)}")

    # Ensure 3-channel RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to MobileNetV2 target resolution
    img_resized = img.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Convert to float numpy array [0, 255]
    img_array = np.array(img_resized, dtype=np.float32)

    # Apply standard MobileNetV2 preprocessing: (pixel / 127.5) - 1.0
    img_preprocessed = (img_array / 127.5) - 1.0

    # Expand dims to create batch dimension (1, 224, 224, 3)
    batch_data = np.expand_dims(img_preprocessed, axis=0)

    return batch_data, img


def _simulate_demo_prediction(img: Image.Image, class_indices: Dict[str, str]) -> List[Tuple[str, float]]:
    """
    Deterministic fallback for presentation and testing when model weights
    have not yet been trained by the user. Analyzes basic image color characteristics
    (green hue, brown spots) to provide an illustrative, educational response.
    """
    num_classes = len(class_indices)
    classes_list = [class_indices[str(i)] for i in range(num_classes)]
    
    # Calculate simple color statistics to determine candidate class
    stat_img = img.resize((50, 50))
    arr = np.array(stat_img, dtype=np.float32)
    r_mean = np.mean(arr[:, :, 0])
    g_mean = np.mean(arr[:, :, 1])
    b_mean = np.mean(arr[:, :, 2])
    
    # Check greenness vs browness
    if g_mean > (r_mean + 15) and g_mean > (b_mean + 15):
        # Green leaf -> healthy tomato or apple
        top_candidates = ["Tomato___healthy", "Apple___healthy", "Pepper__bell___healthy"]
    elif r_mean > g_mean:
        # Brown / orange spots -> rust or early blight
        top_candidates = ["Tomato___Early_blight", "Potato___Early_blight", "Apple___Cedar_apple_rust"]
    else:
        top_candidates = ["Tomato___Late_blight", "Potato___Late_blight", "Grape___Black_rot"]

    # Filter to classes available in class_indices
    valid_candidates = [c for c in top_candidates if c in classes_list]
    if not valid_candidates:
        valid_candidates = [classes_list[0]]

    primary_class = valid_candidates[0]
    primary_conf = 0.8845

    # Find secondary candidate
    secondary_candidates = [c for c in classes_list if c != primary_class]
    sec_class = secondary_candidates[0] if secondary_candidates else primary_class
    sec_conf = 0.0812
    ter_class = secondary_candidates[1] if len(secondary_candidates) > 1 else primary_class
    ter_conf = 0.0343

    return [
        (primary_class, primary_conf),
        (sec_class, sec_conf),
        (ter_class, ter_conf)
    ]


def predict_leaf(file_stream, filename: str = "leaf.jpg") -> Dict[str, Any]:
    """
    Main prediction pipeline:
    1. Validates and preprocesses image.
    2. Runs neural network inference (or demo simulation if weights are pending).
    3. Sorts top-k predictions with calibrated confidence checks.
    4. Enriches with structured recommendations, symptoms, and disclaimers.
    """
    # 1. Validation and Preprocessing
    batch_data, pil_img = validate_and_preprocess_image(file_stream)
    class_indices = load_class_indices()
    model = load_model()

    is_demo = False
    demo_notice = None

    if model is not None:
        try:
            # 2. Real Model Inference
            raw_predictions = model.predict(batch_data, verbose=0)[0]
            top_indices = np.argsort(raw_predictions)[::-1][:3]
            top_preds = [
                (class_indices.get(str(idx), f"Class_{idx}"), float(raw_predictions[idx]))
                for idx in top_indices
            ]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    else:
        # Check if DEMO_MODE is allowed in environment
        demo_allowed = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
        if not demo_allowed:
            raise RuntimeError(
                "Trained model not found at 'backend/models/plant_disease_mobilenetv2.keras'. "
                "Please train the model using 'python train_model.py' before making predictions."
            )
        is_demo = True
        demo_notice = (
            "DEMO MODE: Model weights have not been trained yet. Showing illustrative result. "
            "To train your own model on the PlantVillage dataset, run 'python train_model.py'."
        )
        top_preds = _simulate_demo_prediction(pil_img, class_indices)

    # 3. Analyze top prediction
    top_class, top_conf = top_preds[0]
    plant_name, disease_name = parse_class_name(top_class)
    is_healthy = "healthy" in top_class.lower()

    # Determine confidence status
    is_low_confidence = top_conf < LOW_CONFIDENCE_THRESHOLD
    if is_low_confidence:
        status = "uncertain"
        status_label = "Uncertain / Low Confidence"
    elif is_healthy:
        status = "healthy"
        status_label = "Healthy Leaf"
    else:
        status = "diseased"
        status_label = "Potentially Diseased"

    # 4. Enrich with Agricultural Knowledge Base
    recommendations = get_recommendation(top_class, confidence=top_conf)

    # Format top 3 predictions for UI display
    top_alternatives = []
    for cls_name, conf in top_preds:
        p_name, d_name = parse_class_name(cls_name)
        top_alternatives.append({
            "class_raw": cls_name,
            "plant": p_name,
            "disease": d_name,
            "confidence_pct": round(conf * 100, 2),
            "confidence": round(conf, 4)
        })

    # Prepare response payload
    result = {
        "success": True,
        "is_demo_mode": is_demo,
        "demo_notice": demo_notice,
        "filename": filename,
        "plant_type": plant_name,
        "disease_name": disease_name,
        "status": status,
        "status_label": status_label,
        "confidence": round(top_conf, 4),
        "confidence_pct": round(top_conf * 100, 2),
        "is_low_confidence": is_low_confidence,
        "raw_class": top_class,
        "top_predictions": top_alternatives,
        "recommendations": recommendations,
        "image_meta": {
            "width": pil_img.width,
            "height": pil_img.height,
            "mode": pil_img.mode
        },
        "disclaimer": (
            "AI predictions are assistive decision-support tools and are not a substitute for "
            "professional on-site agronomic diagnosis or laboratory tissue testing. Always adhere "
            "to local agricultural extension advice and chemical label mandates."
        )
    }

    return result
