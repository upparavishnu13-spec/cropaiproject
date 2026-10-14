"""
PlantCare AI - Backend & ML API Test Suite
Tests health check, prediction, validation, error handling, database history,
and agricultural knowledge base mapping.
"""

import os
import io
import json
import pytest
from PIL import Image

# Add backend directory to sys.path
import sys
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app
from database import init_db, clear_history, get_statistics
from recommendations import DISEASE_KNOWLEDGE_BASE, get_recommendation, parse_class_name


@pytest.fixture
def client():
    """Configures a Flask test client with an isolated temporary SQLite database."""
    test_db_path = os.path.join(os.path.dirname(__file__), "test_plantcare.db")
    app.config["TESTING"] = True
    os.environ["DATABASE_PATH"] = test_db_path
    init_db(test_db_path)

    with app.test_client() as client:
        yield client

    # Cleanup temporary test database
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass


def create_synthetic_image(color=(34, 139, 34), format="JPEG", size=(224, 224)):
    """Helper to generate an in-memory synthetic image byte stream."""
    img = Image.new("RGB", size, color=color)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format=format)
    byte_arr.seek(0)
    return byte_arr


# ==========================================
# TEST 1: Health & Classes Endpoints
# ==========================================
def test_health_check(client):
    """Verifies that the /api/health endpoint returns 200 with operational metadata."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "database" in data
    assert data["service"] == "PlantCare AI API"


def test_supported_classes(client):
    """Verifies that /api/classes returns a non-empty list of supported classes."""
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_classes"] > 0
    assert "classes" in data
    assert "grouped_by_plant" in data
    assert "Tomato" in data["grouped_by_plant"]


def test_disease_library_endpoint(client):
    """Verifies that /api/library returns comprehensive knowledge base profiles."""
    response = client.get("/api/library")
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_entries"] >= 20
    first = data["diseases"][0]
    assert "plant" in first
    assert "disease" in first
    assert "symptoms" in first
    assert "immediate_actions" in first


# ==========================================
# TEST 2: Image Prediction & Error Handling
# ==========================================
def test_predict_valid_image(client):
    """Verifies that uploading a valid leaf image returns structured prediction and care advice."""
    img_bytes = create_synthetic_image(color=(40, 160, 40))
    data = {
        "image": (img_bytes, "test_leaf.jpg")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    assert res_json["success"] is True
    assert "plant_type" in res_json
    assert "disease_name" in res_json
    assert "status" in res_json
    assert "confidence" in res_json
    assert "recommendations" in res_json
    assert "top_predictions" in res_json
    assert len(res_json["top_predictions"]) > 0


def test_predict_missing_file(client):
    """Verifies that request without an image file triggers a 400 Bad Request."""
    response = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "No image file provided" in data["error"]


def test_predict_invalid_extension(client):
    """Verifies that uploading an unsupported file format (.txt) is rejected."""
    fake_txt = io.BytesIO(b"Not an image")
    data = {
        "image": (fake_txt, "notes.txt")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Unsupported file type" in data["error"]


def test_predict_corrupt_image(client):
    """Verifies that corrupt or malformed image bytes trigger a clean 400 error."""
    corrupt_bytes = io.BytesIO(b"\xFF\xD8\xFF\xE0" + b"garbage data that cannot decode")
    data = {
        "image": (corrupt_bytes, "corrupt.jpg")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Corrupt or unreadable" in data["error"]


# ==========================================
# TEST 3: Database Scan History & Statistics
# ==========================================
def test_history_and_clear(client):
    """Verifies scan history persistence, statistics calculation, and clearing."""
    # 1. Clear any initial state
    client.post("/api/history/clear")

    # 2. Upload an image to record a scan
    img_bytes = create_synthetic_image()
    client.post("/api/predict", data={"image": (img_bytes, "test.jpg")}, content_type="multipart/form-data")

    # 3. Fetch history
    res = client.get("/api/history")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["statistics"]["total_scans"] == 1
    assert len(data["recent_scans"]) == 1
    assert data["recent_scans"][0]["filename"] == "test.jpg"

    # 4. Clear history
    clear_res = client.post("/api/history/clear")
    assert clear_res.status_code == 200

    # 5. Verify empty
    res_after = client.get("/api/history")
    assert res_after.get_json()["statistics"]["total_scans"] == 0
    assert len(res_after.get_json()["recent_scans"]) == 0


# ==========================================
# TEST 4: Knowledge Base Consistency
# ==========================================
def test_recommendations_consistency():
    """Verifies that all knowledge base entries contain required agronomic fields."""
    assert len(DISEASE_KNOWLEDGE_BASE) >= 20

    for class_key, profile in DISEASE_KNOWLEDGE_BASE.items():
        assert "plant" in profile
        assert "disease" in profile
        assert "is_healthy" in profile
        assert "severity" in profile
        assert "symptoms" in profile and len(profile["symptoms"]) > 0
        assert "causes" in profile and len(profile["causes"]) > 0
        assert "immediate_actions" in profile and len(profile["immediate_actions"]) > 0
        assert "preventative_measures" in profile and len(profile["preventative_measures"]) > 0
        assert "expert_consultation" in profile

    # Test parser
    plant, disease = parse_class_name("Tomato___Early_blight")
    assert plant == "Tomato"
    assert disease == "Early blight"

    # Test fallback on low confidence
    fallback = get_recommendation("Tomato___Early_blight", confidence=0.45)
    assert "candidate_prediction" in fallback
    assert fallback["severity"] == "Uncertain"
