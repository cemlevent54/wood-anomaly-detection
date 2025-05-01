from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api.app.services.efficientad.efficientnet_service import EfficientAdService
from enum import Enum
from PIL import Image
import io

router = APIRouter()

class MODEL_NAME(str, Enum):
    EFFICIENT_AD = "efficient_ad"
    STPM = "stpm"
    PBAS = "pbas"

model_services = {
    MODEL_NAME.EFFICIENT_AD: EfficientAdService(),
    # MODEL_NAME.STPM: STPMService(),
    # MODEL_NAME.PBAS: PBASService(),
}

@router.post("/test_model")
async def test_model(
    model_name: MODEL_NAME = Form(...),
    file: UploadFile = File(...)
):
    service = model_services.get(model_name)
    if not service:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        service.prepare_model_for_testing()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model yükleme hatası: {str(e)}")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prediction, score, anomaly_map_base64 = service.test_model_with_photo(image, show_plot=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test hatası: {str(e)}")

    return {
        "model": model_name,
        "prediction": prediction,
        "score": round(score, 4),
        "anomaly_map_base64": anomaly_map_base64
    }

