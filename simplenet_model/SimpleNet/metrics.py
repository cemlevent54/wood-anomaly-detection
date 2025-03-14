"""Anomaly metrics."""
import cv2
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    
    
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    precision, recall, _ = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auc_pr = metrics.auc(recall, precision)
    
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    # ✅ Maskeleri `np.array` formatına çevir, `None` olanları sıfır yap
    ground_truth_masks = [np.zeros((256, 256), dtype=np.uint8) if mask is None else np.array(mask, dtype=np.uint8) for mask in ground_truth_masks]
    
    # ✅ Maskeleri ve segmentasyonları `256x256` boyutuna getir
    ground_truth_masks = np.stack([cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) for mask in ground_truth_masks])
    anomaly_segmentations = np.stack([cv2.resize(seg, (256, 256), interpolation=cv2.INTER_NEAREST) for seg in anomaly_segmentations])

    # ✅ Bellek tüketimini azaltmak için `int8` formatına çevir
    ground_truth_masks = ground_truth_masks.astype(np.int8).flatten()
    anomaly_segmentations = anomaly_segmentations.astype(np.float32).flatten()

    # ✅ Eğer tüm maskeler sıfırsa, `roc_curve()` hesaplanamaz. Varsayılan değer döndür.
    if np.sum(ground_truth_masks) == 0:
        return {"auroc": 0.0, "fpr": [], "tpr": [], "optimal_threshold": 0.0, "optimal_fpr": 0.0, "optimal_fnr": 1.0}

    # ✅ AUROC hesapla
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth_masks, anomaly_segmentations)
    auroc = metrics.roc_auc_score(ground_truth_masks, anomaly_segmentations)

    # ✅ Precision-Recall eğrisi hesapla
    precision, recall, thresholds = metrics.precision_recall_curve(ground_truth_masks, anomaly_segmentations)
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > ground_truth_masks)
    fnr_optim = np.mean(predictions < ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


import pandas as pd
from skimage import measure
import pandas as pd
import numpy as np
import cv2
from skimage import measure
from sklearn import metrics

def compute_pro(masks, amaps, num_th=200):
    """
    Computes the PRO (Per Region Overlap) metric for anomaly detection.

    Args:
        masks (numpy array): Ground truth segmentation masks (NxHxW).
        amaps (numpy array): Anomaly segmentation maps (NxHxW).
        num_th (int): Number of thresholds for evaluation.

    Returns:
        float: PRO AUC score.
    """
    df = pd.DataFrame(columns=["pro", "fpr", "threshold"])
    
    # ✅ Fazladan boyutları kaldır
    masks = np.squeeze(masks)  # (141,256,256,256) → (141,256,256)
    amaps = np.squeeze(amaps)  # (141,256,256,256) → (141,256,256)

    # ✅ Eğer 4 boyutlu veri gelirse, son boyutu at (RGB gibi durumları düzelt)
    if masks.ndim == 4:
        masks = masks[:, :, :, 0]
    if amaps.ndim == 4:
        amaps = amaps[:, :, :, 0]

    # ✅ Boyut uyuşmazlıklarını kontrol et
    if masks.shape != amaps.shape:
        print(f"UYARI: Boyut uyuşmazlığı! masks: {masks.shape}, amaps: {amaps.shape}")
        target_shape = (masks.shape[1], masks.shape[2])  # (H, W)
        amaps = np.array([cv2.resize(amap, target_shape) for amap in amaps])

    # ✅ İkili maskeleri başlat
    binary_amaps = np.zeros_like(amaps, dtype=np.uint8)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)

            # ✅ Bölgesel analiz yap
            labeled_mask = measure.label(mask)
            for region in measure.regionprops(labeled_mask):
                if region.area > 0:  # 0'a bölünmeyi önle
                    tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                    pros.append(tp_pixels / region.area)

        # ✅ `inverse_masks` ile boyut uyuşmazlığını düzelt
        inverse_masks = 1 - masks
        if inverse_masks.shape != binary_amaps.shape:
            print(f"UYARI: inverse_masks boyutu uyumsuz! {inverse_masks.shape} != {binary_amaps.shape}")
            binary_amaps = np.array([cv2.resize(amap, (inverse_masks.shape[1], inverse_masks.shape[2])) for amap in binary_amaps])

        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum() if inverse_masks.sum() > 0 else 0

        df.loc[len(df)] = {"pro": np.mean(pros) if pros else 0, "fpr": fpr, "threshold": th}

    # ✅ Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    if df["fpr"].max() > 0:
        df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"]) if not df.empty else 0
    return pro_auc
