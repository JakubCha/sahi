# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320


MMDET_YOLOX_TINY_MODEL_URL = (
    "https://huggingface.co/fcakyon/mmdet-yolox-tiny/resolve/main/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
)
MMDET_YOLOX_TINY_MODEL_PATH = "tests/data/models/mmdet-yolox-tiny/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
MMDET_YOLOX_TINY_CONFIG_URL = "https://huggingface.co/fcakyon/mmdet-yolox-tiny/raw/main/yolox_tiny_8x8_300e_coco.py"
MMDET_YOLOX_TINY_CONFIG_PATH = "tests/data/models/mmdet-yolox-tiny/yolox_tiny_8x8_300e_coco.py"
IMAGE_PATH = "tests/data/small-vehicles1.jpeg"


def download_mmdet_yolox_tiny_model():
    download_from_url(MMDET_YOLOX_TINY_MODEL_URL, MMDET_YOLOX_TINY_MODEL_PATH)
    download_from_url(MMDET_YOLOX_TINY_CONFIG_URL, MMDET_YOLOX_TINY_CONFIG_PATH)


class TestMmdetDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.mmdet import MmdetDetectionModel

        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(mmdet_detection_model.model, None)

    def test_perform_inference_with_mask_output(self):
        from sahi.models.mmdet import MmdetDetectionModel

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        boxes = original_predictions[0][0]
        masks = original_predictions[0][1]

        # ensure all prediction scores are greater then 0.5
        for box in boxes[0]:
            if len(box) == 5:
                if box[4] > 0.5:
                    break

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [377, 273, 410, 314])
        self.assertEqual(len(boxes), 80)
        self.assertEqual(len(masks), 80)

    def test_perform_inference_without_mask_output(self):
        from sahi.models.mmdet import MmdetDetectionModel

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image

        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        boxes = original_predictions[0]

        # find box of first car detection with conf greater than 0.5
        for box in boxes[2]:
            print(len(box))
            if len(box) == 5:
                if box[4] > 0.5:
                    break

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [320, 323, 380, 365])
        self.assertEqual(len(boxes), 80)

    def test_convert_original_predictions_with_mask_output(self):
        from sahi.models.mmdet import MmdetDetectionModel

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions(full_shape=(image.shape[0], image.shape[1]))
        object_predictions = mmdet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_predictions), 3)
        self.assertEqual(object_predictions[0].category.id, 2)
        self.assertEqual(object_predictions[0].category.name, "car")
        self.assertEqual(
            object_predictions[0].bbox.to_xywh(),
            [448, 308, 41, 36],
        )
        self.assertEqual(object_predictions[2].category.id, 2)
        self.assertEqual(object_predictions[2].category.name, "car")
        self.assertEqual(
            object_predictions[2].bbox.to_xywh(),
            [381, 280, 33, 30],
        )
        for object_prediction in object_predictions:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions_without_mask_output(self):
        from sahi.models.mmdet import MmdetDetectionModel

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_predictions = mmdet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_predictions), 2)
        self.assertEqual(object_predictions[0].category.id, 2)
        self.assertEqual(object_predictions[0].category.name, "car")
        np.testing.assert_almost_equal(object_predictions[0].bbox.to_xywh(), [320.28, 323.55, 60.60, 41.91], decimal=1)
        self.assertEqual(object_predictions[1].category.id, 2)
        self.assertEqual(object_predictions[1].category.name, "car")
        np.testing.assert_almost_equal(object_predictions[1].bbox.to_xywh(), [448.45, 310.97, 44.49, 30.86], decimal=1)
        for object_prediction in object_predictions:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    def test_perform_inference_without_mask_output_with_automodel(self):
        from sahi import AutoDetectionModel

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = AutoDetectionModel.from_pretrained(
            model_type="mmdet",
            model_path=MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        boxes = original_predictions[0]

        # find box of first car detection with conf greater than 0.5
        for box in boxes[2]:
            print(len(box))
            if len(box) == 5:
                if box[4] > 0.5:
                    break

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [320, 323, 380, 365])
        self.assertEqual(len(boxes), 80)


if __name__ == "__main__":
    unittest.main()
