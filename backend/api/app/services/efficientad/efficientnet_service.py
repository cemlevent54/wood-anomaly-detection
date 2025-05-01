from api.app.services.interfaces.model_interface import ModelInterface
import torch
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from api.app.services.efficientad.common import get_autoencoder, get_pdn_small, get_pdn_medium

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image as PILImage
from PIL import ImageOps
import cv2


class EfficientAdService(ModelInterface):
    
    def log_message(self, message, level="INFO"):
        print(f"[{level}]: {message}")
    
    def __init__(self):
        self.image_size=256
        self.out_channels = 384
        self.model_size = "medium"
        self.model_paths = {
            "teacher": "api/app/modelfiles/efficientad/teacher_final.pth",
            "student": "api/app/modelfiles/efficientad/student_final.pth",
            "autoencoder": "api/app/modelfiles/efficientad/autoencoder_final.pth"
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        
    
    def check_if_model_paths_exist(self):
        for model_name, model_path in self.model_paths.items():
            if not os.path.exists(model_path):
                self.log_message(f"{model_name} modeli bulunamadı: {model_path}", level="ERROR")
                raise FileNotFoundError(f"{model_name} modeli eksik! {model_path} dosyasını kontrol et.")
    
    def load_model_weights(self, path, model_name):
        try:
            self.log_message(f"Model {self.device.upper()} cihazında çalışacak.")
            
            # PyTorch 2.6'daki güvenlik değişiklikleri için 'weights_only=False' açıkça belirtildi.
            state_dict = torch.load(path, map_location=self.device, weights_only=False)

            # Eğer model Sequential olarak kaydedilmişse
            if isinstance(state_dict, torch.nn.Sequential):
                self.log_message(f"{model_name} Sequential modeli olarak doğrudan yüklendi.")
                return state_dict.to(self.device).eval()

            # Eğer sadece state_dict kaydedilmişse
            elif isinstance(state_dict, dict):
                self.log_message(f"{model_name} ağırlıkları state_dict olarak yüklendi.")
                return state_dict

            else:
                self.log_message(f"{model_name} için beklenmeyen dosya formatı: {type(state_dict)}", level="ERROR")
                raise TypeError(f"Geçersiz model formatı: {type(state_dict)}")

        except Exception as e:
            self.log_message(f"{model_name} model yükleme hatası: {str(e)}", level="ERROR")
            raise e
    
    
    def prepare_model_for_testing(self):
        try:
            if self.model_size == "small":
                teacher = get_pdn_small(self.out_channels)
                student = get_pdn_small(2 * self.out_channels)
            elif self.model_size == "medium":
                teacher = get_pdn_medium(self.out_channels)
                student = get_pdn_medium(2 * self.out_channels)
            autoencoder = get_autoencoder(self.out_channels)
            self.log_message("Modeller başarıyla oluşturuldu.")
        except Exception as e:
            self.log_message(f"Model oluşturma hatası: {str(e)}", level="ERROR")
            raise e
        
        teacher_weights = self.load_model_weights(self.model_paths["teacher"], "teacher")
        student_weights = self.load_model_weights(self.model_paths["student"], "student")
        autoencoder_weights = self.load_model_weights(self.model_paths["autoencoder"], "autoencoder")
        
        # Eğer state_dict() olarak yüklenmişse, modellerin içine yükle
        if isinstance(teacher_weights, dict):
            teacher.load_state_dict(teacher_weights)
            teacher.to(self.device).eval()
        if isinstance(student_weights, dict):
            student.load_state_dict(student_weights)
            student.to(self.device).eval()
        if isinstance(autoencoder_weights, dict):
            autoencoder.load_state_dict(autoencoder_weights)
            autoencoder.to(self.device).eval()

        self.log_message("Tüm modeller başarıyla yüklendi ve test için hazır!")
        
        # Teacher normalization için veri gerekir (train_loader yoksa dummy image ile yapılabilir)
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder

        # Dummy ile teacher normalization (alternatif: sabit değer yükle)
        dummy = torch.zeros(1, 3, 256, 256).to(self.device)
        with torch.no_grad():
            t_out = teacher(dummy)
            self.teacher_mean = t_out.mean(dim=[0, 2, 3], keepdim=True)
            self.teacher_std = t_out.std(dim=[0, 2, 3], keepdim=True)
            
            self.teacher_std[self.teacher_std == 0] = 1e-6
    
    def calculate_adaptive_threshold(self, anomaly_map: np.ndarray, method="mean_std", std_coeff=3):
        
        if method == "mean_std":
            mean = np.mean(anomaly_map)
            std = np.std(anomaly_map)
            return mean + std_coeff * std
        elif method == "percentile":
            return np.percentile(anomaly_map, 99)  # Üst %1'lik değer
        else:
            raise ValueError(f"Geçersiz threshold yöntemi: {method}")

    
    def test_model_with_photo(self, image_input, threshold=0.3, show_plot=True):
        if isinstance(image_input, str):
            self.log_message(f"Test görüntüsü: {image_input}")
            image_pil = Image.open(image_input).convert("RGB")
        else:
            image_pil = image_input

        score, anomaly_map, original_size = self.predict_single_image(image_pil)

        # 🔽 Threshold'u dış fonksiyondan al
        if threshold is None:
            threshold = self.calculate_adaptive_threshold(anomaly_map, method="mean_std", std_coeff=3)

        if math.isinf(score) or math.isnan(score):
            score = 9999.0
            
        prediction = "Anomali" if score > threshold else "Normal"

        self.log_message(f"Tahmin: {prediction} | Skor: {score:.4f}")

        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image_pil)
            plt.title("Orijinal Görsel")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(anomaly_map, cmap='jet')  # Anomali haritası renkli
            plt.title(f"Anomaly Map\nSkor: {score:.3f}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        anomaly_base64 = self.anomaly_map_to_base64(anomaly_map, original_size)

        return prediction, score, anomaly_base64

    
    def anomaly_map_to_base64(self, anomaly_map: np.ndarray, original_size) -> str:
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        norm_map = (norm_map * 255).astype(np.uint8)
        
        # Resize to match input image size (width, height)
        resized = cv2.resize(norm_map, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap
        colored_map = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB for correct display
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        # Convert to PIL image and base64 encode
        img_pil = Image.fromarray(colored_map)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"


   
    @torch.no_grad()
    def predict_single_image(self, image_pil):
        original_size = image_pil.size  # (width, height)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        teacher_out = self.teacher(image_tensor)
        teacher_out = (teacher_out - self.teacher_mean) / self.teacher_std

        student_out = self.student(image_tensor)
        ae_out = self.autoencoder(image_tensor)

        map_st = torch.mean((teacher_out - student_out[:, :self.out_channels]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((ae_out - student_out[:, self.out_channels:]) ** 2, dim=1, keepdim=True)
        
        map_combined = 0.5 * map_st + 0.5 * map_ae
        anomaly_score = map_combined.max().item()

        # Anomaly map for visualization
        anomaly_map = map_combined.squeeze().cpu().numpy()
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        norm_map = (norm_map * 255).astype(np.uint8)

        # OpenCV ile resize (PIL.Image.size = (width, height))
        resized_map = cv2.resize(norm_map, original_size, interpolation=cv2.INTER_LINEAR)


        return anomaly_score, resized_map, original_size
    
    