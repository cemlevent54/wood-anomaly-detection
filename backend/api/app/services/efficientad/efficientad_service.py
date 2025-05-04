from PIL import Image
import torch
import math

from api.app.services.interfaces.model_interface import ModelInterface
from api.app.services.efficientad.transforms import get_default_transforms
from api.app.services.efficientad.utils import calculate_adaptive_threshold, remove_border_artifacts, draw_filtered_contours
import tifffile
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
import cv2
import os
from pathlib import Path


from api.app.services.efficientad.common import get_autoencoder, get_pdn_small, get_pdn_medium

#=========== visualizer.py imports ===========#

import io
import base64
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np

#=========== visualizer.py imports ===========#

class EfficientAdService(ModelInterface):
    def __init__(self):
        self.image_size = 256
        self.out_channels = 384
        self.model_size = "medium"
        self.model_paths = {
            "teacher": "api/app/modelfiles/efficientad/teacher_final.pth",
            "student": "api/app/modelfiles/efficientad/student_final.pth",
            "autoencoder": "api/app/modelfiles/efficientad/autoencoder_final.pth",
            # opsiyonel
            "quantiles": "api/app/modelfiles/efficientad/quantiles.pth"
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = get_default_transforms(self.image_size)

        self.log_message("Model yükleniyor...")

        # 🔸 Model tanımlamaları
        if self.model_size == "small":
            teacher = get_pdn_small(self.out_channels)
            student = get_pdn_small(2 * self.out_channels)
        elif self.model_size == "medium":
            teacher = get_pdn_medium(self.out_channels)
            student = get_pdn_medium(2 * self.out_channels)
        else:
            raise ValueError(f"Geçersiz model boyutu: {self.model_size}")

        autoencoder = get_autoencoder(self.out_channels)

        # 🔸 Model ağırlıklarını yükle
        for model, key in zip([teacher, student, autoencoder], ["teacher", "student", "autoencoder"]):
            path = self.model_paths[key]
            if not os.path.exists(path):
                self.log_message(f"{key} modeli bulunamadı: {path}", level="ERROR")
                raise FileNotFoundError(f"{key} modeli eksik! {path}")

            loaded = torch.load(path, map_location=self.device)
            if isinstance(loaded, dict):
                self.log_message(f"{key} state_dict olarak yüklendi.")
                model.load_state_dict(loaded)
            elif isinstance(loaded, torch.nn.Module):
                self.log_message(f"{key} doğrudan model olarak yüklendi.")
                model = loaded
            else:
                raise TypeError(f"{key} için geçersiz model formatı: {type(loaded)}")

            model.to(self.device).eval()
            if key == "teacher":
                teacher = model
            elif key == "student":
                student = model
            elif key == "autoencoder":
                autoencoder = model
            
            self.teacher = teacher
            self.student = student
            self.autoencoder = autoencoder

        # 🔸 Opsiyonel: Quantile dosyasını yükle
        quantiles = None
        quantile_path = self.model_paths.get("quantiles")
        if quantile_path and os.path.exists(quantile_path):
            try:
                self.log_message(f"Quantile dosyası bulunuyor: {quantile_path}")
                quantiles = torch.load(quantile_path, map_location=self.device)
            except Exception as e:
                self.log_message(f"Quantile dosyası yüklenirken hata: {e}", level="ERROR")

        

        self.q_st_start = quantiles.get("q_st_start") if quantiles else None
        self.q_st_end = quantiles.get("q_st_end") if quantiles else None
        self.q_ae_start = quantiles.get("q_ae_start") if quantiles else None
        self.q_ae_end = quantiles.get("q_ae_end") if quantiles else None

        self.log_message("Model hazır.")


    def log_message(self, message, level="INFO"):
        print(f"[{level}]: {message}")
        
    
    def test_model_with_photo(self, image_input, threshold=None, show_plot=True):
        self.log_message("UYARI: test_model_with_photo() kullanılmıyor, lütfen test_image() fonksiyonunu kullanın.", level="WARNING")
        return self.test_image(image_input)  # direkt yönlendirme
    
    @torch.no_grad()
    def _predict_on_image_tensor(self, image_tensor, original_size):
        teacher_out = self.teacher(image_tensor)
        if not hasattr(self, 'teacher_mean') or not hasattr(self, 'teacher_std') or self.teacher_mean is None:
            self.teacher_mean = teacher_out.mean(dim=[0, 2, 3], keepdim=True)
            self.teacher_std = teacher_out.std(dim=[0, 2, 3], keepdim=True)
            self.teacher_std[self.teacher_std == 0] = 1e-6

        teacher_out = (teacher_out - self.teacher_mean) / self.teacher_std
        student_out = self.student(image_tensor)
        ae_out = self.autoencoder(image_tensor)

        map_st = torch.mean((teacher_out - student_out[:, :self.out_channels]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((ae_out - student_out[:, self.out_channels:]) ** 2, dim=1, keepdim=True)

        if self.q_st_start is not None and self.q_st_end is not None:
            map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start + 1e-6)
        if self.q_ae_start is not None and self.q_ae_end is not None:
            map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start + 1e-6)

        map_combined = 0.5 * map_st + 0.5 * map_ae
        score = map_combined.max().item()

        anomaly_map = map_combined.squeeze().cpu().numpy()
        norm_map = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)) * 255
        resized_map = cv2.resize(norm_map.astype(np.uint8), original_size, interpolation=cv2.INTER_LINEAR)

        return score, resized_map

    
    def test_image(self, image_pil, threshold=None):
        try:
            # 🔸 Threshold belirleme
            if threshold is None:
                threshold = 0.3
            self.log_message(f"Threshold: {threshold}")

            # 🔸 Tahmin yap — quantile parametreleri çıkarıldı
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            score, anomaly_map = self._predict_on_image_tensor(image_tensor, image_pil.size)
            
            # 🔸 Anomali haritasını temizle
            cleaned_map = remove_border_artifacts(anomaly_map.copy(), border_px=30)
            
            # Anomaly haritasını normalize et (uint8 0-255 aralığında)
            norm_map = (cleaned_map - cleaned_map.min()) / (cleaned_map.max() - cleaned_map.min() + 1e-6)
            norm_map_uint8 = (norm_map * 255).astype(np.uint8)

            prediction = "Anomali" if score > threshold else "Normal"

            

            # 🔸 Görselleştirme
            anomaly_map_base64 = anomaly_map_to_base64(norm_map_uint8, image_pil.size)
            overlay_np = draw_filtered_contours(np.array(image_pil), norm_map_uint8, threshold=0.35)
            overlay_base64 = overlay_to_base64(overlay_np)

            # 🔸 GT mask varsa metrikleri hesapla
            image_name = getattr(image_pil, "filename", None)
            image_stem = Path(image_name).stem if image_name else "input"
            possible_paths = [
                f"api/app/images/wood/ground_truth/defect/{image_stem}_mask.png",
                f"api/app/images/wood/ground_truth/defect/{image_stem}_mask.jpg",
                f"api/app/images/wood/ground_truth/defect/{image_stem}_mask.tif"
            ]
            gt_mask_path = next((p for p in possible_paths if os.path.exists(p)), None)

            f1 = iou = None
            if gt_mask_path:
                self.log_message(f"GT mask bulundu: {gt_mask_path}")
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, cleaned_map.shape[::-1])
                gt_binary = (gt_mask > 127).astype(np.uint8)
                pred_binary = (norm_map_uint8 > threshold * 255).astype(np.uint8)

                debug_overlay = np.stack([gt_binary * 255, pred_binary * 255, np.zeros_like(gt_binary)], axis=-1)
                debug_path = f"debug_overlay_{image_stem}.png"
                # cv2.imwrite(debug_path, debug_overlay)

                f1 = f1_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
                iou = jaccard_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
            else:
                self.log_message(f"GT mask BULUNAMADI. Kontrol edilen yollar: {possible_paths}", level="WARNING")

            return {
                "model": "efficient_ad",
                "prediction": prediction,
                "score": round(score, 4),
                "anomaly_map_base64": anomaly_map_base64,
                "overlay_base64": overlay_base64,
                "f1_score": round(f1, 4) if f1 is not None else None,
                "iou_score": round(iou, 4) if iou is not None else None
            }

        except Exception as e:
            raise RuntimeError(f"EfficientAd model hatası: {str(e)}")


#=========== visualizer.py ===========#




def log_message(message, level="INFO"):
    print(f"[{level}]: {message}")

def plot_results(image_pil, anomaly_map, score):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_pil)
    plt.title("Orijinal Görsel")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(anomaly_map, cmap='jet')
    plt.title(f"Anomaly Map\nSkor: {score:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def anomaly_map_to_base64(anomaly_map, original_size):
    if anomaly_map.dtype != np.uint8:
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

    resized = cv2.resize(anomaly_map, original_size)
    colored = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
def overlay_contours(original_image, anomaly_map, threshold=0.5):
    normalized_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    mask = (normalized_map > threshold).astype(np.uint8) * 255

    # Gürültü azaltmak için morfolojik açma
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Küçük konturları filtrele
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]

    original_cv = np.array(original_image)
    if original_cv.ndim == 2:
        original_cv = cv2.cvtColor(original_cv, cv2.COLOR_GRAY2BGR)
    elif original_cv.shape[2] == 4:
        original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGBA2RGB)

    overlay = original_cv.copy()
    cv2.drawContours(overlay, filtered, -1, (0, 0, 255), thickness=2)
    return overlay


def overlay_to_base64(overlay_np):
    import io
    from PIL import Image
    import base64
    import numpy as np

    if overlay_np.dtype != np.uint8:
        overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)

    if overlay_np.ndim == 2:
        overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_GRAY2RGB)

    img_pil = Image.fromarray(overlay_np)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def visualize_examples(test_paths, transform, teacher, student, autoencoder, device, threshold):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    sample_images = np.random.choice(test_paths, min(5, len(test_paths)), replace=False)
    for img_path in sample_images:
        try:
            image = Image.open(img_path).convert("RGB")
            size = image.size
            image_tensor = transform(image).unsqueeze(0).to(device)

            from api.app.services.efficientad.efficientad_service import EfficientAdService
            service = EfficientAdService()
            service.teacher = teacher
            service.student = student
            service.autoencoder = autoencoder
            anomaly_map = service._predict_on_image_tensor(image_tensor, size)[1]
            
            score = np.max(anomaly_map)
            prediction = "Anomali" if score > threshold else "Normal"

            overlayed = overlay_contours(image, anomaly_map, threshold)

            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Tahmin: {prediction} (Skor: {score:.4f})", fontsize=14)

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Orijinal Görüntü")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(anomaly_map, cmap="jet")
            plt.title("Anomali Haritası")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlayed)
            plt.title("Konturlu Görsel")
            plt.axis("off")

            plt.show()
        except Exception as e:
            log_message(f"Görselleştirme hatası: {img_path} - {str(e)}", level="ERROR")


def numpy_to_base64(np_array, format="JPEG"):
    """
    NumPy array (H x W x C) → Base64 string (image/jpeg format default)
    """
    image = PILImage.fromarray(np_array.astype("uint8"))
    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded