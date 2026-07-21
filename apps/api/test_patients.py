from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

response = client.post(
    "/api/v1/patients/",
    json={"name": "John Doe", "age": 30, "gender": "M", "weight": 70.5, "height": 175.0, "medical_history": {"diabetes": False}, "vital_signs": {"hr": 70}}
)
print("POST /api/v1/patients/ response:")
print(json.dumps(response.json(), indent=2))

patient_id = response.json().get("id")
if patient_id:
    get_response = client.get(f"/api/v1/patients/{patient_id}")
    print("\nGET /api/v1/patients/{id} response:")
    print(json.dumps(get_response.json(), indent=2))
