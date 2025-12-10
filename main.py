# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# CORS (útil durante desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Haar cascade (opcional, mantenemos como fallback)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def prepare_image(contents, target_size=(640, 480)):
    # lee bytes -> numpy array -> BGR image
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen inválida")
    # Redimensionar manteniendo proporción mínima (acelerar)
    img = cv2.resize(img, target_size)
    return img

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    img = prepare_image(contents, target_size=(640, 480))

    # convertir a gris y ecualizar histograma para mejorar contraste
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # detectar con HaarCascade (mejorando parámetros)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    result = []
    for (x, y, w, h) in faces:
        result.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return {"count": len(result), "faces": result}
