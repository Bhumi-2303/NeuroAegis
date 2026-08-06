import requests
import time

job_id = 'db4150c6-eb54-42c5-abab-b92bbd635aca'
response = requests.get(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}")
print(response.json())
