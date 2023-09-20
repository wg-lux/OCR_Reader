import requests
import json

url = 'http://127.0.0.1:5000/api_endpoint'

# List of file paths
image_paths = ['testfiles/01-01.pdf', 'testfiles/02-01.pdf', 'testfiles/02-02.pdf', 'testfiles/03-01.pdf', 'testfiles/03-02.pdf']

# Your coordinates, for example:
coordinates = [
    {'left': 0.0001, 'top': 0.4, 'right': 1, 'bottom': 0.9},
    # ... more coordinates for other pages
]

# Initialize an empty list to store responses
responses = []

for image_path in image_paths:
    files = {'image': open(image_path, 'rb')}
    
    # Use the first set of coordinates for the first page, and no coordinates for the second page
    if image_paths.index(image_path) == 0:
        data = {'coordinates': json.dumps(coordinates)}
    else:
        data = {}  # No coordinates for cropping
    
    response = requests.post(url, files=files, data=data)
    
    # Check for errors in the response
    if response.status_code >= 400:
        print(f"Error for {image_path}: {response.status_code} - {response.text}")
    else:
        try:
            # Try to parse the response as JSON
            data = response.json()
            print(f"Response for {image_path}: {data}")
            responses.append(data)
        except requests.exceptions.JSONDecodeError as e:
            print(f"Failed to decode JSON response for {image_path}: {e}")

# Now, 'responses' contains the JSON data for each file's response
