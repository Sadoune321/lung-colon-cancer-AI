import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf   

MODEL_PATH = "best_vit_lc25000"
IMG_SIZE = 224  

CLASSES = {
    0: "colon_aca  → Colon Adenocarcinoma",
    1: "colon_n    → Colon Benign Tissue",
    2: "lung_aca   → Lung Adenocarcinoma",
    3: "lung_n     → Lung Benign Tissue",
    4: "lung_scc   → Lung Squamous Cell Carcinoma",
}

app = FastAPI(title="LC25000 ViT API", version="1.0")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modèle chargé !")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.expand_dims(arr, 0)


@app.get("/")
def root():
    return {"message": "LC25000 ViT API — POST /predict avec une image"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modèle non chargé"})

    contents = await file.read()
    tensor = preprocess(contents)

    proba = model(tensor, training=False).numpy()[0]

    pred_idx = int(np.argmax(proba))
    pred_name = CLASSES[pred_idx]
    confidence = float(proba[pred_idx])

    return {
        "filename": file.filename,
        "predicted_class": pred_name,
        "confidence": round(confidence, 4),
        "all_probs": {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }