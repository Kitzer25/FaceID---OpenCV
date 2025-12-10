from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

app = FastAPI()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    result = []
    for (x, y, w, h) in faces:
        result.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return {
        "count": len(faces),
        "faces": result
    }
