import io
import base64
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np


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

            from .predictor import test_model  # test_model başka dosyadaysa

            anomaly_map = test_model(image_tensor, teacher, student, autoencoder, device, size)
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