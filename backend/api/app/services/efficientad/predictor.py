import torch
import numpy as np
import cv2

class AnomalyPredictor:
    def __init__(self, teacher, student, autoencoder, out_channels, device):
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.out_channels = out_channels
        self.device = device

        self.teacher_mean = None
        self.teacher_std = None

    @torch.no_grad()
    def predict(
        self, image_tensor, original_size,
        q_st_start=None, q_st_end=None,
        q_ae_start=None, q_ae_end=None
    ):
        teacher_out = self.teacher(image_tensor)
        if self.teacher_mean is None or self.teacher_std is None:
            # Eğer dışarıdan set edilmediyse dummy input üzerinden ortalama/std hesapla
            self.teacher_mean = teacher_out.mean(dim=[0, 2, 3], keepdim=True)
            self.teacher_std = teacher_out.std(dim=[0, 2, 3], keepdim=True)
            self.teacher_std[self.teacher_std == 0] = 1e-6

        teacher_out = (teacher_out - self.teacher_mean) / self.teacher_std
        student_out = self.student(image_tensor)
        ae_out = self.autoencoder(image_tensor)

        # Anomali haritalarını hesapla
        map_st = torch.mean((teacher_out - student_out[:, :self.out_channels]) ** 2, dim=1, keepdim=True)
        map_ae = torch.mean((ae_out - student_out[:, self.out_channels:]) ** 2, dim=1, keepdim=True)

        # Opsiyonel normalizasyon (efficientad.py ile uyumlu)
        if q_st_start is not None and q_st_end is not None:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start + 1e-6)
        if q_ae_start is not None and q_ae_end is not None:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start + 1e-6)

        map_combined = 0.5 * map_st + 0.5 * map_ae
        score = map_combined.max().item()

        # Görsel harita hazırlığı
        anomaly_map = map_combined.squeeze().cpu().numpy()
        norm_map = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)) * 255
        resized_map = cv2.resize(norm_map.astype(np.uint8), original_size, interpolation=cv2.INTER_LINEAR)

        return score, resized_map
