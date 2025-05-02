import numpy as np
import cv2

def calculate_adaptive_threshold(anomaly_map, method="mean_std", std_coeff=3):
    if method == "mean_std":
        return np.mean(anomaly_map) + std_coeff * np.std(anomaly_map)
    elif method == "percentile":
        return np.percentile(anomaly_map, 99)
    raise ValueError(f"Geçersiz threshold yöntemi: {method}")

def remove_border_artifacts(anomaly_map, border_px=30):
    h, w = anomaly_map.shape
    mask = np.zeros_like(anomaly_map)
    mask[border_px:h-border_px, border_px:w-border_px] = 1
    return anomaly_map * mask

def filter_contours(contours, image_shape, min_area=100, border_margin=30):
    h, w = image_shape[:2]
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        if area > min_area and border_margin < x < w-border_margin and border_margin < y < h-border_margin:
            filtered.append(cnt)
    return filtered

def draw_filtered_contours(image_np, anomaly_map, threshold=None):
    if anomaly_map is None or anomaly_map.size == 0:
        raise ValueError("draw_filtered_contours: anomaly_map boş.")

    if anomaly_map.dtype != np.uint8:
        anomaly_map = ((anomaly_map - anomaly_map.min()) /
                       (anomaly_map.max() - anomaly_map.min() + 1e-8)) * 255
        anomaly_map = anomaly_map.astype(np.uint8)

    if threshold is None:
        threshold_val = calculate_adaptive_threshold(anomaly_map, method="percentile", std_coeff=3)
    else:
        threshold_val = threshold * 255

    mask = (anomaly_map > threshold_val).astype(np.uint8) * 255
    if mask is None or mask.sum() == 0:
        return image_np  # hiçbir kontur yoksa orijinal resmi döndür

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours, image_np.shape, min_area=30, border_margin=5)

    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    overlay = image_np.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=2)

    return overlay


