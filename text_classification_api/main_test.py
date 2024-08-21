from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_classify_positive():
    response = client.post("/classify", json={"text": "I love this movie!"})
    assert response.status_code == 200
    assert "class" in response.json()
    assert "confidence" in response.json()

def test_classify_negative():
    response = client.post("/classify", json={"text": "I hate this movie."})
    assert response.status_code == 200
    assert "class" in response.json()
    assert "confidence" in response.json()

def test_invalid_input():
    response = client.post("/classify", json={"invalid": "input"})
    assert response.status_code == 422