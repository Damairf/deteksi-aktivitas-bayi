import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI()

model = YOLO("best_aktivitas.pt")

@app.get("/")
def home():
    return {"status": "API Server Aktif 24/7 di Vercel"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    results = model(frame, conf=0.60)

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        cls_id = int(box.cls[0])
        status = results[0].names[cls_id]
        conf = float(box.conf[0])
    else:
        status = "Tidak Terdeteksi"
        conf = 0.0

    return {
        "status_aktivitas": status,
        "confidence": round(conf, 2)
    }