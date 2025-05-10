# stpm/files/test.py
import torch
# import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import argparse

# Basit mean ve std tanımları (ayrıca servis tarafında da tanımlı olmalı)
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

# Argparse yerine doğrudan namespace kullanımı (hparams için)
args = argparse.Namespace(
    input_size=256,
    amap_mode='mul',
    load_size=256
)

# Basit min-max normalizasyonu
def min_max_norm(image):
    a_min, a_max = np.percentile(image, 1), np.percentile(image, 99)
    return np.clip((image - a_min) / (a_max - a_min + 1e-6), 0, 1)

# Model sınıfı
class STPM(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model_t = resnet18(pretrained=True).eval()
        self.model_s = resnet18(pretrained=False)
        
        self.features_t = []
        self.features_s = []

        # Hook'lar için
        def hook_t(m, i, o): self.features_t.append(o)
        def hook_s(m, i, o): self.features_s.append(o)

        self.model_t.layer1[-1].register_forward_hook(hook_t)
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)

        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)

    def init_features(self):
        self.features_t = []
        self.features_s = []

    def forward(self, x):
        self.features_t.clear()
        self.features_s.clear()
        _ = self.model_t(x)
        _ = self.model_s(x)
        return self.features_t, self.features_s

    def cal_anomaly_map(self, fs_list, ft_list, out_size=256):
        anomaly_map = np.ones((out_size, out_size))

        for ft, fs in zip(ft_list, fs_list):
            ft_n = F.normalize(ft, p=2)
            fs_n = F.normalize(fs, p=2)
            amap = 1 - F.cosine_similarity(fs_n, ft_n)
            amap = F.interpolate(amap.unsqueeze(1), size=out_size, mode='bilinear')[0, 0].cpu().numpy()
            anomaly_map *= amap
        return anomaly_map
