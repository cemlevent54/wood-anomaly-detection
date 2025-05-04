from api.app.services.interfaces.model_interface import ModelInterface
import torch
import os
import copy

from api.app.services.uninet.Uninet_lib.resnet import wide_resnet50_2
from api.app.services.uninet.Uninet_lib.de_resnet import de_wide_resnet50_2
from .files.datasets import loading_dataset
from api.app.services.uninet.Uninet_lib.DFS import DomainRelated_Feature_Selection
from .files.utils import load_weights, t2np, to_device
from api.app.services.uninet.Uninet_lib.model import UniNet
from .files.eval import evaluation_indusAD, evaluation_batch, evaluation_mediAD, evaluation_polypseg
from api.app.services.uninet.Uninet_lib.mechanism import weighted_decision_mechanism
from scipy.ndimage import gaussian_filter

import cv2
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.nn import functional as F
import numpy as np
import base64



class Config:
    dataset = "MVTec AD"
    setting = "oc"
    domain = "industrial"
    _class_ = "wood"
    image_size = 256
    batch_size = 1
    T = 2
    weighted_decision_mechanism = True
    alpha = 0.01
    beta = 0.00003

c = Config()


class UninetService(ModelInterface):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_message("Model yükleniyor...")

        self.config = Config()
        c = self.config
        dataset_name = c.dataset
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_path = os.path.join(BASE_DIR, "modelfiles", "uninet")
        
        dataset_info = loading_dataset(c, dataset_name)
        self.test_dataloader = dataset_info[1]

        Source_teacher, bn = wide_resnet50_2(c, pretrained=True)
        Source_teacher.layer4 = None
        Source_teacher.fc = None
        student = de_wide_resnet50_2(pretrained=False)
        DFS = DomainRelated_Feature_Selection()
        [Source_teacher, bn, student, DFS] = to_device([Source_teacher, bn, student, DFS], self.device)
        Target_teacher = copy.deepcopy(Source_teacher)

        suffix = 'BEST_P_PRO'
        new_state = load_weights([Target_teacher, bn, student, DFS], ckpt_path, suffix)
        self.model = UniNet(
            c,
            Source_teacher.to(self.device).eval(),
            new_state['tt'],
            new_state['bn'],
            new_state['st'],
            new_state['dfs']
        )

    def log_message(self, message, level="INFO"):
        print(f"[{level}]: {message}")

    def test_model_with_photo(self, image_input, threshold=None, show_plot=True):
        self.log_message("UYARI: test_model_with_photo() kullanılmıyor, lütfen test_image() fonksiyonunu kullanın.", level="WARNING")
        return self.test_image(image_input, threshold=threshold)

    def test_image(self, image_pil, mask_pil=None, threshold=None):
        try:
            c = self.config
            model = self.model
            device = self.device

            original_image = image_pil.convert("RGB")
            orig_width, orig_height = original_image.size

            transform = T.Compose([
                T.Resize((c.image_size, c.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(original_image).unsqueeze(0).to(device)

            model.train_or_eval(type='eval')
            n = model.n
            output_list = [list() for _ in range(n * 3)]

            with torch.no_grad():
                t_tf, de_features = model(input_tensor)
                for l, (t, s) in enumerate(zip(t_tf, de_features)):
                    output = 1 - F.cosine_similarity(t, s)
                    output_list[l].append(output)

                _, anomaly_map = weighted_decision_mechanism(1, output_list, c.alpha, c.beta)
                anomaly_map = gaussian_filter(anomaly_map, sigma=4).squeeze()

            # Resize to original size
            anomaly_map_resized = cv2.resize(anomaly_map, (orig_width, orig_height))
            anomaly_map_norm = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            threshold_value = np.percentile(anomaly_map_norm, 95)
            _, anomaly_mask_bin = cv2.threshold(anomaly_map_norm, threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(anomaly_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Orijinal RGB görüntüyü numpy array'e çevir
            # original_rgb = np.array(original_image)
            # # RGB -> BGR: OpenCV'nin anlayacağı hale getir
            # original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
            # # BGR görüntü üzerinde konturları çiz
            # contour_image_bgr = cv2.drawContours(original_bgr.copy(), contours, -1, (0, 0, 255), 3)  # kırmızı kontur
            # # Sonuç görselini tekrar RGB'ye döndür
            # contour_image_rgb = cv2.cvtColor(contour_image_bgr, cv2.COLOR_BGR2RGB)
            
            
            original_rgb = np.array(original_image)
            contour_image = cv2.drawContours(original_rgb.copy(), contours, -1, (255, 0, 0), 3)

            # IoU
            iou_score = None
            if mask_pil is not None:
                gt_mask = np.array(mask_pil.resize((orig_width, orig_height)).convert('L'))
                pred_mask = (anomaly_mask_bin > 0).astype(np.uint8)
                gt_mask_bin = (gt_mask > 127).astype(np.uint8)
                intersection = np.logical_and(pred_mask, gt_mask_bin).sum()
                union = np.logical_or(pred_mask, gt_mask_bin).sum()
                iou_score = intersection / union if union != 0 else 0.0

            # Base64 encode
            def encode(img_arr, cmap=None):
                if cmap:
                    img_arr = cv2.applyColorMap(img_arr, cmap)
                _, buffer = cv2.imencode('.jpg', img_arr)
                return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            result = {
                "model": "uninet",
                "prediction": "anomalous" if anomaly_map_resized.max() > (threshold or 0.5) else "normal",
                "score": round(float(anomaly_map_resized.max()), 4),
                "anomaly_map_base64": encode(anomaly_map_norm, cmap=cv2.COLORMAP_JET),
                "overlay_base64": encode(contour_image),
                "f1_score": None,
                "iou_score": round(iou_score, 4) if iou_score is not None else None
            }
            return result

        except Exception as e:
            self.log_message(f"Hata oluştu: {str(e)}", level="ERROR")
            raise
