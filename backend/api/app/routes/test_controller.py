
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api.app.services.efficientad.efficientad_service import EfficientAdService
from api.app.services.uninet.uninet_service import UninetService
from api.app.services.revisiting_reverse_dissilation.revisiting_reverse_dissilation_service import RevisitingReverseDissilationService
from api.app.services.stpm.stpm_service import STPMService
# from api.app.services.stpm.stpm_service import STPMService
# from api.app.services.pbas.pbas_service import PBASService
from enum import Enum
from PIL import Image
import io

router = APIRouter()


class MODEL_NAME(str, Enum):
    EFFICIENT_AD = "efficient_ad"
    # STPM = "stpm"
    # PBAS = "pbas"
    REVISITING_REVERSE_DISSILATION = "revisiting_reverse_dissilation"
    UNINET = "uninet"
    STPM = "stpm"
    


model_services = {
    MODEL_NAME.EFFICIENT_AD: EfficientAdService(),
    # MODEL_NAME.STPM: STPMService(),
    # MODEL_NAME.PBAS: PBASService(),
    MODEL_NAME.REVISITING_REVERSE_DISSILATION: RevisitingReverseDissilationService(),
    MODEL_NAME.UNINET: UninetService(),
    MODEL_NAME.STPM: STPMService(),  # STPMService()  # Uncomment when STPMService is available
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
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.filename = file.filename
        
        result = service.test_image(image)  # 👈 tek noktadan yönetim

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test hatası: {str(e)}")

    return result
