import torch
import numpy as np

# Custom transforms
class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image.transpose(2, 0, 1)).float()

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        return (image - self.mean) / self.std
    

## ========= utils_train.py =========

import torch
import torch.nn as nn
from torch.nn import functional as F
import geomloss

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//4, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
    def forward(self, x):
        return self.proj(x)
    
class MultiProjectionLayer(nn.Module):
    def __init__(self, base = 64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)
    def forward(self, features, features_noise = False):
        if features_noise is not False:
            return ([self.proj_a(features_noise[0]),self.proj_b(features_noise[1]),self.proj_c(features_noise[2])], \
                  [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])])
        else:
            return [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])]

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


class CosineReconstruct(nn.Module):
    def __init__(self):
        super(CosineReconstruct, self).__init__()
    def forward(self, x, y):
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))

class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, consistent_shuffle = True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                              reach=None, diameter=10000000, scaling=0.95, \
                                truncate=10, cost=None, kernel=None, cluster_scale=None, \
                                  debias=True, potentials=False, verbose=False, backend='auto')
        self.reconstruct = CosineReconstruct()       
        self.contrast = torch.nn.CosineEmbeddingLoss(margin = 0.5)
    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """
        current_batchsize = projected_normal_feature[0].shape[0]

        target = -torch.ones(current_batchsize).to('cuda')

        normal_proj1 = projected_normal_feature[0]
        normal_proj2 = projected_normal_feature[1]
        normal_proj3 = projected_normal_feature[2]
        # shuffling samples order for caculating pair-wise loss_ssot in batch-mode , (for efficient computation)
        shuffle_index = torch.randperm(current_batchsize)
        # Shuffle the feature order of samples in each block
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature
        loss_ssot = self.sinkhorn(torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1), torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1),-1),  torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1),-1),  torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1),-1))
        loss_reconstruct = self.reconstruct(abnormal_proj1, normal_proj1)+ \
                   self.reconstruct(abnormal_proj2, normal_proj2)+ \
                   self.reconstruct(abnormal_proj3, normal_proj3)
        loss_contrast = self.contrast(noised_feature1.view(noised_feature1.shape[0], -1), normal_proj1.view(normal_proj1.shape[0], -1), target = target) +\
                           self.contrast(noised_feature2.view(noised_feature2.shape[0], -1), normal_proj2.view(normal_proj2.shape[0], -1), target = target) +\
                           self.contrast(noised_feature3.view(noised_feature3.shape[0], -1), normal_proj3.view(normal_proj3.shape[0], -1), target = target)
        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast)/1.11


## ========= utils_test.py =========

import torch
from torch.nn import functional as F
import cv2
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from statistics import mean
from scipy.ndimage import gaussian_filter
import warnings




warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



# def evaluation_multi_proj(encoder,proj,bn, decoder, dataloader,device):
#     encoder.eval()
#     proj.eval()
#     bn.eval()
#     decoder.eval()
#     gt_list_px = []
#     pr_list_px = []
#     gt_list_sp = []
#     pr_list_sp = []
#     aupro_list = []
#     with torch.no_grad():
#         for (img, gt, label, _, _) in dataloader:

#             img = img.to(device)
#             inputs = encoder(img)
#             features = proj(inputs)
#             outputs = decoder(bn(features))
#             anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
#             anomaly_map = gaussian_filter(anomaly_map, sigma=4)
#             gt[gt > 0.5] = 1
#             gt[gt <= 0.5] = 0
#             if label.item()!=0:
#                 aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
#                                               anomaly_map[np.newaxis,:,:]))
#             gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
#             pr_list_px.extend(anomaly_map.ravel())
#             gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
#             pr_list_sp.append(np.max(anomaly_map))

#         auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
#         auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
#     return auroc_px, auroc_sp, round(np.mean(aupro_list),4)

def evaluation_multi_proj(encoder, proj, bn, decoder, dataloader, device, class_name=None, save_heatmaps=False):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for (img, gt, label, _, filename) in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # Heatmap kaydet
            if save_heatmaps and class_name is not None:
                # anomaly_map [H, W] → [1, H, W]
                anomaly_tensor = torch.tensor(anomaly_map).unsqueeze(0)
                save_dir = os.path.join('./RD++_checkpoint_result', class_name, 'heatmaps')
                save_path = os.path.join(save_dir, filename[0].replace('.png', '.jpg').replace('.tif', '.jpg'))
                save_heatmap_on_image(img[0].cpu(), anomaly_tensor, save_path)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item() != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))


        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),4)





def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

### my codes for heatmap

import cv2
import numpy as np
import os
import torch

def generate_anomaly_map(original_features, reconstructed_features):
    maps = []
    for of, rf in zip(original_features, reconstructed_features):
        cos = torch.nn.functional.cosine_similarity(of, rf, dim=1)
        anomaly_map = 1 - cos
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        maps.append(anomaly_map.unsqueeze(1))
    return torch.mean(torch.cat(maps, dim=1), dim=1, keepdim=True)

def save_heatmap_on_image(image_tensor, anomaly_map, save_path):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
    image_np = image_np.astype(np.uint8)

    heatmap = anomaly_map.squeeze().cpu().numpy()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)




