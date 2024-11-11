from fastapi import FastAPI, File, UploadFile

from src.core.detector import get_slot_occupancy
from src.blob.blob_manager import blob_manager


app = FastAPI()

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    image_name = "parking_lot_image.jpg"
    image_data = await file.read()
    blob_manager.upload_image(image_data, image_name)
    return {"message": "Image uploaded successfully"}

@app.get("/get_parking_lot_state/")
def get_parking_lot_state(model_version = 'v8'):
    image_name = "parking_lot_image.jpg"
    try:
        image_data = blob_manager.load_image(image_name)
        
        return get_slot_occupancy(image_data, model_version=model_version)
    except Exception as e:
        return {"error": str(e)}
