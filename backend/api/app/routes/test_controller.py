from fastapi import APIRouter
from api.app.services.efficientnet_service import EfficientNetService
# Diğer modelleri burada import edebilirsin

router = APIRouter()

# Servis örneğini oluştur
model_services = [
    EfficientNetService(),
    # STPMService(), PBASService() vs.
]

@router.get("/test_model_with_photo")
def test_model():
    results = []
    for service in model_services:
        results.append(service.test_model_with_photo())
    return {"results": results}
