# from .visualizer import anomaly_map_to_base64
# from .predictor import AnomalyPredictor
# from .transforms import get_default_transforms
# import torch

# def run_test_on_single_image(
#     image_pil,
#     teacher,
#     student,
#     autoencoder,
#     teacher_mean,
#     teacher_std,
#     q_st_start=None,
#     q_st_end=None,
#     q_ae_start=None,
#     q_ae_end=None,
#     return_map=False
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     transform = get_default_transforms(256)
#     image_tensor = transform(image_pil).unsqueeze(0).to(device)
#     size = image_pil.size

#     # AnomalyPredictor örneği
#     predictor = AnomalyPredictor(teacher, student, autoencoder, out_channels=384, device=device)
#     predictor.teacher_mean = teacher_mean
#     predictor.teacher_std = teacher_std

#     # Sadece 2 parametre!
#     score, anomaly_map = predictor.predict(
#         image_tensor, size,
#         q_st_start=q_st_start, q_st_end=q_st_end,
#         q_ae_start=q_ae_start, q_ae_end=q_ae_end
#     )

#     if return_map:
#         return score, anomaly_map
#     else:
#         base64_map = anomaly_map_to_base64(anomaly_map, size)
#         return score, base64_map
