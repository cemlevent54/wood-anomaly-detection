import torch
import os
from api.app.services.efficientad.common import get_autoencoder, get_pdn_small, get_pdn_medium


class ModelLoader:
    def __init__(self, model_size, out_channels, device, model_paths, log):
        self.model_size = model_size
        self.out_channels = out_channels
        self.device = device
        self.model_paths = model_paths
        self.log = log

    def check_model_paths(self):
        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                self.log(f"{name} modeli bulunamadı: {path}", level="ERROR")
                raise FileNotFoundError(f"{name} modeli eksik! {path}")

    def load_state_dict(self, path, model_name):
        self.log(f"Model {self.device.upper()} cihazında çalışacak.")
        loaded = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(loaded, dict):
            self.log(f"{model_name} state_dict olarak yüklendi.")
            return "state_dict", loaded
        elif isinstance(loaded, torch.nn.Module):
            self.log(f"{model_name} doğrudan model olarak yüklendi (Sequential vb).")
            return "model", loaded
        else:
            raise TypeError(f"{model_name} için geçersiz model formatı: {type(loaded)}")

    def load_models(self):
        if self.model_size == "small":
            teacher = get_pdn_small(self.out_channels)
            student = get_pdn_small(2 * self.out_channels)
        elif self.model_size == "medium":
            teacher = get_pdn_medium(self.out_channels)
            student = get_pdn_medium(2 * self.out_channels)
        else:
            raise ValueError(f"Geçersiz model boyutu: {self.model_size}")

        autoencoder = get_autoencoder(self.out_channels)

        # Öğrenilmiş ağırlıkları yükle
        for model, key in zip([teacher, student, autoencoder], ["teacher", "student", "autoencoder"]):
            mode, loaded = self.load_state_dict(self.model_paths[key], key)
            if mode == "state_dict":
                model.load_state_dict(loaded)
            elif mode == "model":
                model = loaded
            model.to(self.device).eval()
            if key == "teacher":
                teacher = model
            elif key == "student":
                student = model
            elif key == "autoencoder":
                autoencoder = model

        # Quantile dosyası varsa yükle, yoksa None döndür
        quantiles = None
        quantile_path = self.model_paths.get("quantiles")
        if quantile_path:
            try:
                if os.path.exists(quantile_path):
                    self.log(f"Quantile dosyası bulunuyor: {quantile_path}")
                    quantiles = torch.load(quantile_path, map_location=self.device)
                else:
                    self.log("Quantile dosyası bulunamadı, normalizasyon yapılmayacak.", level="WARNING")
            except Exception as e:
                self.log(f"Quantile dosyası yüklenirken hata oluştu: {e}", level="ERROR")
                quantiles = None

        return teacher, student, autoencoder, quantiles


