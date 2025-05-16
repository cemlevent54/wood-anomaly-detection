from api.app.services.interfaces.model_interface import ModelInterface
import io
import base64
import torch
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import transforms
from torch.nn import functional as F

# STPM model ve args importu
from api.app.services.stpm.files.test import STPM, args

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def min_max_norm(image):
    a_min, a_max = np.percentile(image, 1), np.percentile(image, 99)
    return np.clip((image - a_min) / (a_max - a_min + 1e-6), 0, 1)

class STPMService(ModelInterface):
    def __init__(self, model_ckpt_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, "..", "..", "modelfiles", "stpm", "epoch=99-step=299.ckpt")
        checkpoint_path = os.path.normpath(checkpoint_path)
        self.log_message(f"Model yükleme yolu: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
        self.model_ckpt_path = checkpoint_path
        
        self.model = self._load_model(self.model_ckpt_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=mean_train, std=std_train),
        ])
        self.inv_normalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean_train, std_train)],
            std=[1/s for s in std_train]
        )

    def _load_model(self, ckpt_path):
        model = STPM(hparams=args)
        if ckpt_path:
            # model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
        return model.to(self.device)

    def log_message(self, message, level="INFO"):
        print(f"[{level}]: {message}")

    def test_model_with_photo(self, image_input, threshold=None, show_plot=True):
        self.log_message("UYARI: test_model_with_photo() kullanılmıyor, lütfen test_image() fonksiyonunu kullanın.", level="WARNING")
        return self.test_image(image_input, threshold=threshold)

    def test_image(self, image_input, threshold=0.6):
        # Resmi oku ve modele uygun hale getir
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError("Unsupported image input format.")
        
        original_width, original_height = image.size
        
        image_resized = image.resize((256, 256))

        input_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            # Çıktıyı unpack etme: Eğer birden fazla çıktı varsa
            if isinstance(output, tuple):
                features_t, features_s = output
            else:
                # Eğer tek bir çıktı dönüyorsa
                features_t = features_s = output
            
            anomaly_map = self.model.cal_anomaly_map(features_s, features_t, out_size=256)

        input_tensor_inv = self.inv_normalize(input_tensor.squeeze()).permute(1, 2, 0).cpu().numpy()
        input_tensor_inv = np.clip(input_tensor_inv * 255, 0, 255).astype(np.uint8)

        # Heatmap ve kontur işlemleri
        anomaly_map_norm = min_max_norm(anomaly_map)
        heatmap = cv2.applyColorMap(np.uint8(anomaly_map_norm * 255), cv2.COLORMAP_JET)
        binary_map = (anomaly_map_norm > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Konturların çizilmesi, ince ama belirgin çizgiler
        contoured = input_tensor_inv.copy()
        cv2.drawContours(contoured, contours, -1, (255, 0, 0), 1)  # Kontur çizim kalınlığını 1'e düşürme
        contoured_bgr = cv2.cvtColor(contoured, cv2.COLOR_RGB2BGR)
        
        # Görüntüyü orijinal boyutuna geri çevirme
        contoured_resized = cv2.resize(contoured_bgr, (original_width, original_height))

        # Heatmap'in de başlangıç boyutlarına göre yeniden boyutlandırılması
        heatmap_resized = cv2.resize(heatmap, (original_width, original_height))

        # Base64 encode for the contoured image
        _, buffer_contoured = cv2.imencode('.jpg', contoured_resized)
        img_bytes_contoured = buffer_contoured.tobytes()
        img_base64_contoured = base64.b64encode(img_bytes_contoured).decode('utf-8')

        # Base64 encode for the resized heatmap
        _, buffer_heatmap = cv2.imencode('.jpg', heatmap_resized)
        img_bytes_heatmap = buffer_heatmap.tobytes()
        img_base64_heatmap = base64.b64encode(img_bytes_heatmap).decode('utf-8')

        # Return both heatmap and contoured image in a JSON-like format
        return {
            "model": "stpm",
            "anomaly_map_base64": f"data:image/jpeg;base64,{img_base64_heatmap}",
            "overlay_base64": f"data:image/jpeg;base64,{img_base64_contoured}",
            "results": {
                "f1_score": 0.8333,
                "iou_score": 0.7158,
                "pixel_level_auc_roc": 0.9490476032437287,
                "total_image_level_auc_roc": 0.8961770623742454
            }
        }