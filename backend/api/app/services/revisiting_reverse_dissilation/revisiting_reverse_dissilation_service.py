from api.app.services.interfaces.model_interface import ModelInterface
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score
from scipy.ndimage import gaussian_filter

from api.app.services.revisiting_reverse_dissilation.model.resnet import wide_resnet50_2
from api.app.services.revisiting_reverse_dissilation.model.de_resnet import de_wide_resnet50_2
from api.app.services.revisiting_reverse_dissilation.utils import ToTensor, Normalize, MultiProjectionLayer, cal_anomaly_map

import os
from pathlib import Path

class RevisitingReverseDissilationService(ModelInterface):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_message("Model yükleniyor...")

        # === Model tanımı ===
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False)
        proj_layer = MultiProjectionLayer(base=64)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, "..", "..", "modelfiles", "revisitingreversedissilation", "wres50_wood.pth")
        checkpoint_path = os.path.normpath(checkpoint_path)

        # === Checkpoint yükleme ===
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )
 
        proj_layer.load_state_dict(checkpoint['proj'])
        decoder.load_state_dict(checkpoint['decoder'])
        bn.load_state_dict(checkpoint['bn'])
        
        

        self.encoder = encoder.to(self.device).eval()
        self.bn = bn.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.proj_layer = proj_layer.to(self.device).eval()

        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.log_message("Model hazır.")

    def log_message(self, message, level="INFO"):
        print(f"[{level}]: {message}")

    def test_model_with_photo(self, image_input, threshold=None, show_plot=True):
        self.log_message("UYARI: test_model_with_photo() kullanılmıyor, lütfen test_image() fonksiyonunu kullanın.", level="WARNING")
        return self.test_image(image_input, threshold=threshold)

    def test_image(self, image_pil, mask_pil=None, threshold=None):
        try:
            if threshold is None:
                threshold = 0.75
            self.log_message(f"Threshold: {threshold}")

            # === 1. Preprocess ===
            img_np = np.array(image_pil) / 255.0
            orig_h, orig_w = img_np.shape[:2]
            img_resized_np = cv2.resize(img_np, (256, 256))
            img_tensor = self.transform(img_resized_np).unsqueeze(0).to(self.device)

            # === 2. Inference ===
            with torch.no_grad():
                features = self.encoder(img_tensor)
                proj_features = self.proj_layer(features)
                decoded_features = self.decoder(self.bn(proj_features))

                decoded_features = [
                    F.interpolate(df, size=f.shape[-2:], mode='bilinear', align_corners=True)
                    if df.shape[-2:] != f.shape[-2:] else df
                    for df, f in zip(decoded_features, features)
                ]

                anomaly_map, _ = cal_anomaly_map(features, decoded_features, out_size=256, amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # === 3. Postprocess ===
            heatmap = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            binary_mask = (heatmap_resized > threshold).astype(np.uint8)

            # === 4. Overlay + Contours ===
            original_image_bgr = cv2.cvtColor(np.uint8(img_np * 255), cv2.COLOR_RGB2BGR)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_image_bgr, contours, -1, (0, 0, 255), 2)
            overlay = original_image_bgr  # 🔴 artık overlay, konturlu orijinal resimdir

            # === 5. Metrikler ===
            f1 = iou = None
            if mask_pil is not None:
                gt_mask = np.array(mask_pil.convert("L").resize((orig_w, orig_h))) > 0
                gt_mask = gt_mask.astype(np.uint8)
                f1 = f1_score(gt_mask.ravel(), binary_mask.ravel())
                iou = jaccard_score(gt_mask.ravel(), binary_mask.ravel())

            # === 6. Base64 encode ===
            def encode(img_array):
                _, buffer = cv2.imencode('.jpg', img_array)
                encoded = base64.b64encode(buffer).decode('utf-8')
                return "data:image/jpeg;base64," + encoded

            anomaly_map_base64 = encode(heatmap_color)
            overlay_base64 = encode(overlay)

            return {
                "model": "revisiting_reverse_dissilation",
                "prediction": "anomalous" if heatmap_resized.max() > threshold else "normal",
                "score": round(float(heatmap_resized.max()), 4),
                "anomaly_map_base64": anomaly_map_base64,
                "overlay_base64": overlay_base64,
                "f1_score": round(f1, 4) if f1 is not None else None,
                "iou_score": round(iou, 4) if iou is not None else None
            }

        except Exception as e:
            self.log_message(f"Model test edilirken hata oluştu: {e}", level="ERROR")
            return None
