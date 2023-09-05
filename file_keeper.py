from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

image_name = "nlp.jpeg"
image_endpoint = image_name.split(".")[0]
image_type = image_name.split(".")[1]
@app.get(f"/{image_endpoint}")
async def get_image():
    return FileResponse(f"images/{image_name}", media_type=f"image/{image_type}")

image_name = "nlp.jpeg"
image_endpoint = image_name.split(".")[0]
image_type = image_name.split(".")[1]
@app.get(f"/{image_endpoint}")
async def get_image():
    return FileResponse(f"images/{image_name}", media_type=f"image/{image_type}")