from PIL import Image
import torch
import math

from api.app.services.interfaces.model_interface import ModelInterface
from api.app.services.efficientad.transforms import get_default_transforms
from api.app.services.efficientad.model_loader import ModelLoader
from api.app.services.efficientad.predictor import AnomalyPredictor
from api.app.services.efficientad.visualizer import plot_results, anomaly_map_to_base64
from api.app.services.efficientad.visualizer import overlay_contours, overlay_to_base64, numpy_to_base64
from api.app.services.efficientad.utils import calculate_adaptive_threshold, remove_border_artifacts, draw_filtered_contours

from api.app.services.efficientad.efficient_test import run_test_on_single_image
import tifffile
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
import cv2
import os
from pathlib import Path




class EfficientAdService(ModelInterface):
    def __init__(self):
        self.image_size = 256
        self.out_channels = 384
        self.model_size = "medium"
        self.model_paths = {
            "teacher": "api/app/modelfiles/efficientad/teacher_final.pth",
            "student": "api/app/modelfiles/efficientad/student_final.pth",
            "autoencoder": "api/app/modelfiles/efficientad/autoencoder_final.pth"
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = get_default_transforms(self.image_size)

        self.log_message("Model yükleniyor...")
        loader = ModelLoader(self.model_size, self.out_channels, self.device, self.model_paths, self.log_message)
        loader.check_model_paths()
        teacher, student, autoencoder, quantiles = loader.load_models()

        self.predictor = AnomalyPredictor(teacher, student, autoencoder, self.out_channels, self.device)
        if quantiles:
            self.q_st_start = quantiles["q_st_start"]
            self.q_st_end = quantiles["q_st_end"]
            self.q_ae_start = quantiles["q_ae_start"]
            self.q_ae_end = quantiles["q_ae_end"]
        else:
            self.q_st_start = None
            self.q_st_end = None
            self.q_ae_start = None
            self.q_ae_end = None
        self.log_message("Model hazır.")

    def log_message(self, message, level="INFO"):
        print(f"[{level}]: {message}")
        
    
    def test_model_with_photo(self, image_input, threshold=None, show_plot=True):
        self.log_message("UYARI: test_model_with_photo() kullanılmıyor, lütfen test_image() fonksiyonunu kullanın.", level="WARNING")
        return self.test_image(image_input)  # direkt yönlendirme
    def test_image(self, image_pil, threshold=None):
        try:
            # 🔸 Threshold belirleme
            if threshold is None:
                threshold = 0.3
            self.log_message(f"Threshold: {threshold}")

            # 🔸 Tahmin yap — quantile parametreleri çıkarıldı
            score, anomaly_map = run_test_on_single_image(
                image_pil=image_pil,
                teacher=self.predictor.teacher,
                student=self.predictor.student,
                autoencoder=self.predictor.autoencoder,
                teacher_mean=self.predictor.teacher_mean,
                teacher_std=self.predictor.teacher_std,
                q_st_start=None,
                q_st_end=None,
                q_ae_start=None,
                q_ae_end=None,
                return_map=True
            )
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

