from gradio_client import Client

client = Client("Lin-Chen/ShareCaptioner-Video")  # Hugging Face Space
result = client.predict(
    video="camp_5min.mp4",   # or a public video URL
    prompt="Describe this video in detail:",
    api_name="/predict"
)
print(result)
