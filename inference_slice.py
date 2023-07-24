from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction

small_image_path = r'F:\DT\Estonia_Birds_2023\Collection_1\inference_toy_small\000\3659_4_1.tif'
big_image_path = r'F:\DT\Estonia_Birds_2023\Collection_2\orthomosaics_sahi\3068.tiff'

# detection_model = Yolov8DetectionModel(model_path='best_detection.pt', device='cuda:0', confidence_threshold=0.3)

# result = get_sliced_prediction(small_image_path, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")
# result = get_sliced_prediction(big_image_path, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")

segmentation_model = Yolov8DetectionModel(model_path='best_segmentation.pt', device='cuda:0', confidence_threshold=0.3)

# result = get_sliced_prediction(small_image_path, segmentation_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")
result = get_sliced_prediction(big_image_path, segmentation_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")

print(result)
