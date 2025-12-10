from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cargar modelos ---
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

mtcnn = MTCNN(image_size=160, margin=14)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def prepare_image(contents):
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen inválida")
    return img, img.shape[1], img.shape[0]

def get_embedding_from_crop(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    aligned = mtcnn(crop_rgb)
    if aligned is None:
        return None
    with torch.no_grad():
        emb = resnet(aligned.unsqueeze(0))
    return emb[0].cpu().numpy().tolist()

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    img, width, height = prepare_image(contents)

    # crear blob para DNN
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    result = []
    conf_threshold = 0.6

    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(width, x2); y2 = min(height, y2)

        # recortar imagen del rostro
        face_crop = img[y1:y2, x1:x2]

        embedding = get_embedding_from_crop(face_crop)

        result.append({
            "box": {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2-x1),
                "h": int(y2-y1),
            },
            "confidence": confidence,
            "embedding": embedding  # puede ser None si no se alineó bien
        })

    return {
        "count": len(result),
        "faces": result
    }
