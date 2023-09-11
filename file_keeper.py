from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

# @app.get(f"/{image_endpoint}")
# async def get_image():
#     return FileResponse(f"images/{image_name}", media_type=f"image/{image_type}")


@app.get("/W1")
async def get_pdf():
    return FileResponse("resources/Generative_AI_with_Large_Language_Models/W1.pdf", media_type=f"application/pdf")

@app.get("/W2")
async def get_pdf():
    return FileResponse("resources/Generative_AI_with_Large_Language_Models/W2.pdf", media_type=f"application/pdf")

@app.get("/W3")
async def get_pdf():
    return FileResponse("resources/Generative_AI_with_Large_Language_Models/W3.pdf", media_type=f"application/pdf")