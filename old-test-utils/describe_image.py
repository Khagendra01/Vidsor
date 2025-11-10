import requests
import base64
import time

# Read the image and encode it to base64
with open("hiker.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Start timing
start_time = time.time()

# Call Ollama API
response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'bakllava',
        'prompt': 'Describe this image in detail.',
        'images': [image_data],
        'stream': False
    }
)

# End timing
end_time = time.time()
inference_time = end_time - start_time

# Print the response
result = response.json()
print(result['response'])
print(f"\nInference time: {inference_time:.2f} seconds")

