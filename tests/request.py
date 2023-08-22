import requests
import json

url = 'http://127.0.0.1:5000/api_endpoint'
image_path = 'testfiles/demo-report.pdf'

# Your coordinates, for example:
coordinates = [
    {'left': 0, 'top': 0.4, 'right': 1, 'bottom': 0.9},
    # ... more coordinates for other pages
]

files = {'image': open(image_path, 'rb')}
data = {'coordinates': json.dumps(coordinates)}  # Convert list of coordinates to JSON string

response = requests.post(url, files=files, data=data)

# Check for errors in the response
if response.status_code >= 400:
    print(f"Error: {response.status_code} - {response.text}")
else:
    try:
        # Try to parse the response as JSON
        data = response.json()
        print(data)
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
