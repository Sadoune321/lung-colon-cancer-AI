import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import timm

MODEL_PATH = "best_vit_lc25000.pth"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = {
    0: "colon_aca  → Colon Adenocarcinoma",
    1: "colon_n    → Colon Benign Tissue",
    2: "lung_aca   → Lung Adenocarcinoma",
    3: "lung_n     → Lung Benign Tissue",
    4: "lung_scc   → Lung Squamous Cell Carcinoma",
}


# ── Architecture EXACTE déduite de l'inspection du checkpoint ─────────────────
#
# Clés backbone : cls_token, pos_embed, patch_embed.*, blocks.*, norm.*
#   → sauvegardé à plat (sans préfixe "backbone.")
#   → on utilise timm directement sans encapsulation
#
# Clés tête :
#   head.0.weight [768]       → LayerNorm(768)
#   head.0.bias   [768]       → LayerNorm(768)
#   head.2.weight [512, 768]  → Linear(768, 512)
#   head.2.bias   [512]
#   head.5.weight [5, 512]    → Linear(512, 5)
#   head.5.bias   [5]
#
# Structure head :
#   head.0 → LayerNorm(768)
#   head.1 → GELU ou ReLU  (pas de poids → pas dans state_dict)
#   head.2 → Linear(768, 512)
#   head.3 → GELU ou ReLU  (pas de poids)
#   head.4 → Dropout       (pas de poids)
#   head.5 → Linear(512, 5)
# ─────────────────────────────────────────────────────────────────────────────
class ViTCustom(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Backbone ViT — on garde les noms de clés timm à plat
        # en utilisant num_classes=0 pour désactiver la tête timm
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        embed_dim = self.vit.num_features  # 768

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),       # head.0  → [768]
            nn.GELU(),                      # head.1  (pas de poids)
            nn.Linear(embed_dim, 512),      # head.2  → [512, 768]
            nn.GELU(),                      # head.3  (pas de poids)
            nn.Dropout(0.1),                # head.4  (pas de poids)
            nn.Linear(512, num_classes),    # head.5  → [5, 512]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


app = FastAPI(title="LC25000 ViT API", version="3.0")
model = None


@app.on_event("startup")
def load_model():
    global model

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remapper les clés :
    # backbone à plat  (cls_token, blocks.X.*, norm.*)  → vit.cls_token, vit.blocks.X.*, vit.norm.*
    # tête             (head.X.*)                        → head.X.*  (inchangé)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            new_state_dict[k] = v
        else:
            new_state_dict["vit." + k] = v

    net = ViTCustom(num_classes=5)
    net.load_state_dict(new_state_dict, strict=True)
    net.to(DEVICE)
    net.eval()
    model = net
    print(f"✅ Modèle chargé sur {DEVICE} !")


def preprocess(image_bytes: bytes) -> torch.Tensor:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img    = img.resize((IMG_SIZE, IMG_SIZE))
    arr    = np.array(img, dtype=np.float32) / 255.0
    arr    = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


@app.get("/")
def root():
    return {"message": "LC25000 ViT API — POST /predict avec une image"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modèle non chargé"})

    contents = await file.read()
    tensor   = preprocess(contents)

    with torch.no_grad():
        logits = model(tensor)
        proba  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx   = int(np.argmax(proba))
    pred_name  = CLASSES[pred_idx]
    confidence = float(proba[pred_idx])

    return {
        "filename":        file.filename,
        "predicted_class": pred_name,
        "confidence":      round(confidence, 4),
        "all_probs":       {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }