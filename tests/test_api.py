import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app

def test_classify_valid():
    with TestClient(app) as client:
        payload = {
            "pairs": [
                {"premise": "I love playing soccer in the rain.", "hypothesis": "Soccer is my favorite sport."},
                {"premise": "The sun is shining bright.", "hypothesis": "It is raining cats and dogs."}
            ]
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert "premise" in pred
            assert "hypothesis" in pred
            assert "label" in pred
            assert pred["label"] in ["entailment", "neutral", "contradiction"]
            assert "confidence" in pred
            assert isinstance(pred["confidence"], float)

def test_classify_malformed():
    with TestClient(app) as client:
        # Missing premise
        payload = {
            "pairs": [
                {"hypothesis": "Only hypothesis here"}
            ]
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 422
        
        # Wrong data type
        payload = {
            "pairs": "not a list of pairs"
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 422

def test_classify_edge_cases():
    with TestClient(app) as client:
        # Empty pairs array
        payload = {
            "pairs": []
        }
        response = client.post("/classify", json=payload)
        # Should be caught by the custom validation raising 400 Bad Request
        assert response.status_code in [400, 422]
        
        # Single-pair batch
        payload = {
            "pairs": [
                {"premise": "A simple sentence.", "hypothesis": "Another simple sentence."}
            ]
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 1

        # Very long text
        long_text = "This is a very long text. " * 100
        payload = {
            "pairs": [
                {"premise": long_text, "hypothesis": "A short hypothesis."}
            ]
        }
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 1
