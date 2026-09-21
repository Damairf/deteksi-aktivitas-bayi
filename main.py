import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO

app = FastAPI(title="Smart Vision - Baby Activity Detection API")

model_aktivitas = YOLO("best_aktivitas.pt")

@app.post("/predict")
async def predict_activity(file: UploadFile = File(...)):
    try:
        # Baca bytes gambar (.webp, .jpg, .png, .jpeg)
        contents = await file.read()
        
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Konversi ke array OpenCV (BGR)
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Predict threshold confidence 0.60
        results = model_aktivitas(frame, conf=0.60)

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            status_aktivitas = results[0].names[cls_id]
            confidence = float(box.conf[0])
        else:
            status_aktivitas = "Tidak Terdeteksi"
            confidence = 0.0

        return {
            "status_aktivitas": status_aktivitas,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Format gambar tidak valid/rusak: {str(e)}")