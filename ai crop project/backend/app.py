"""
PlantCare AI - Flask Backend REST API
Serves disease prediction, historical logs, knowledge base,
and static frontend assets with full CORS and error handling.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from predict import predict_leaf, get_model_status, allowed_file, load_class_indices
from database import init_db, add_scan, get_recent_scans, get_statistics, clear_history
from recommendations import get_all_library_entries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PlantCare-API")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB limit

# Initialize Flask app
app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Configure CORS
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
CORS(app, resources={r"/api/*": {"origins": [o.strip() for o in allowed_origins]}})

# Ensure database is ready
try:
    init_db()
    logger.info("Database initialized successfully.")
except Exception as e:
    logger.error(f"Database initialization error: {e}")


# ==========================================
# STATIC ASSETS & FRONTEND SERVING
# ==========================================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serves index.html and static frontend assets (css, js, images)."""
    if path != "" and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


# ==========================================
# API ENDPOINTS
# ==========================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Reports operational status of the Flask backend, SQLite database,
    and TensorFlow model.
    """
    model_status = get_model_status()
    db_stats = get_statistics()

    return jsonify({
        "status": "healthy",
        "service": "PlantCare AI API",
        "version": "1.0.0",
        "python_version": sys.version.split()[0],
        "model": model_status,
        "database": {
            "status": "connected",
            "total_scans_logged": db_stats["total_scans"]
        }
    }), 200


@app.route("/api/classes", methods=["GET"])
def get_supported_classes():
    """
    Returns the complete list of plant and disease classes
    supported by the model and knowledge base.
    """
    class_indices = load_class_indices()
    classes_list = list(class_indices.values())

    # Group classes by plant species for easy frontend navigation
    plants = {}
    for cls in classes_list:
        if "___" in cls:
            plant_name, disease_name = cls.split("___", 1)
            plant_name = plant_name.replace("_", " ").strip()
            disease_name = disease_name.replace("_", " ").strip()
        else:
            plant_name = "Plant"
            disease_name = cls

        if plant_name not in plants:
            plants[plant_name] = []
        plants[plant_name].append({
            "raw_class": cls,
            "disease_name": disease_name,
            "is_healthy": "healthy" in disease_name.lower()
        })

    return jsonify({
        "total_classes": len(classes_list),
        "classes": classes_list,
        "grouped_by_plant": plants
    }), 200


@app.route("/api/library", methods=["GET"])
def get_disease_library():
    """
    Returns the rich educational disease library containing symptoms,
    causes, immediate care, and organic management for all supported classes.
    """
    entries = get_all_library_entries()
    return jsonify({
        "total_entries": len(entries),
        "diseases": entries
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Leaf Disease Detection Endpoint.
    Accepts multipart/form-data leaf image, runs inference,
    logs diagnostic results to SQLite, and returns structured care advice.
    """
    # 1. Check for file presence
    file_key = "image" if "image" in request.files else ("file" if "file" in request.files else None)
    if not file_key:
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Please upload an image with key 'image' or 'file'."
        }), 400

    file = request.files[file_key]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename. Please select a valid image file."
        }), 400

    # 2. Validate file extension
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed formats: .jpg, .jpeg, .png, .webp, .bmp."
        }), 400

    try:
        # 3. Perform image validation, preprocessing, and inference
        result = predict_leaf(file.stream, filename=file.filename)

        # 4. Record scan in database
        try:
            scan_id = add_scan(
                plant_type=result["plant_type"],
                disease_name=result["disease_name"],
                status=result["status"],
                confidence=result["confidence"],
                raw_class=result["raw_class"],
                filename=result["filename"],
                is_low_confidence=result["is_low_confidence"],
                notes="Analyzed via PlantCare AI"
            )
            result["scan_id"] = scan_id
        except Exception as db_err:
            logger.warning(f"Could not log scan to database: {db_err}")
            result["scan_id"] = None

        return jsonify(result), 200

    except ValueError as val_err:
        # Image corruption or format error
        logger.warning(f"Validation error on image upload: {val_err}")
        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 400
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"An error occurred while analyzing the leaf image: {str(e)}"
        }), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Returns recent scans and aggregate performance statistics
    from the SQLite database.
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        recent_scans = get_recent_scans(limit=limit)
        stats = get_statistics()

        return jsonify({
            "success": True,
            "statistics": stats,
            "recent_scans": recent_scans
        }), 200
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch scan history."
        }), 500


@app.route("/api/history/clear", methods=["POST"])
def clear_scan_history():
    """
    Clears all recorded scans from SQLite after user confirmation.
    """
    try:
        deleted_count = clear_history()
        return jsonify({
            "success": True,
            "message": f"Successfully cleared {deleted_count} scan records."
        }), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to clear scan history."
        }), 500


# ==========================================
# ERROR HANDLERS
# ==========================================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "Uploaded image file is too large. Maximum supported size is 16 MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": f"API endpoint '{request.path}' not found."
        }), 404
    # Fallback to frontend for client-side routing
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error. Please check server logs."
    }), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    print("\n" + "=" * 60)
    print("   🌱 PlantCare AI - Backend Server Starting")
    print(f"   API & Web UI: http://{host}:{port}/")
    print(f"   Health check: http://{host}:{port}/api/health")
    print("=" * 60 + "\n")

    app.run(host=host, port=port, debug=debug)
