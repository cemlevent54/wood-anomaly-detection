from api.app.services.interfaces.model_interface import ModelInterface

class EfficientNetService(ModelInterface):
    def test_model_with_photo(self):
        return {"model": "EfficientNet", "status": "tested"}