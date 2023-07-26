import requests

url = 'http://127.0.0.1:5000/api_endpoint'
image_path = 'testfiles/dateityptest.pdf'
files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)

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
