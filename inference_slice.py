import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from datetime import datetime
import logging
import pickle
from pathlib import Path
from affine import Affine

from osgeo import gdal
from shapely.affinity import affine_transform
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction, predict

small_image_path = r'F:\DT\Estonia_Birds_2023\Collection_1\inference_toy_small\000\3659_4_1.tif'
big_image_path = r'F:\DT\Estonia_Birds_2023\Collection_2\orthomosaics_sahi\test\3068.tif'

# detection_model = Yolov8DetectionModel(model_path='best_detection.pt', device='cuda:0', confidence_threshold=0.3)
# result = get_sliced_prediction(small_image_path, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")
# result = get_sliced_prediction(big_image_path, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")

# segmentation_model = Yolov8DetectionModel(model_path=r'G:\DT\DL\jcharyton\sahi_change_mask_type_repo\sahi\best_segmentation.pt', device='cuda:0', confidence_threshold=0.3)
# result = get_sliced_prediction(small_image_path, segmentation_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")
# result = get_sliced_prediction(big_image_path, segmentation_model, slice_height=640, slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1, postprocess_type="GREEDYNMM")

images_directory = r'F:\DT\Estonia_Birds_2023\Collection_1\inference_toy_small\999'

export_dict = predict(
            model_type='yolov8',
            model_path=r'G:\DT\DL\jcharyton\sahi_change_mask_type_repo\sahi\best_segmentation.pt',
            model_device='cuda:0',
            model_confidence_threshold=0.3,
            source=images_directory,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1, 
            verbose=2,
            novisual=True,
            export_pickle=True,
            return_dict=True
    )

print(export_dict)


def get_image_transformations(orto_path):
    assert Path(orto_path).exists(), f"Error: The file {orto_path} does not exist."
    image_dataset = gdal.Open(orto_path)
    geotransform = image_dataset.GetGeoTransform()
    # convert it to shapely 
    affine_trans_parameters = Affine.from_gdal(*geotransform).to_shapely()

    # get affine_transform parameters
    a = affine_trans_parameters[0]
    b = affine_trans_parameters[1]
    d = affine_trans_parameters[2]
    e = affine_trans_parameters[3]
    x_off = affine_trans_parameters[4]
    y_off = affine_trans_parameters[5]
    
    return a, b, d, e, x_off, y_off

def transform_annotation(annotation, orto_path):
    a, b, d, e, x_off, y_off = get_image_transformations(str(orto_path))
    annotation_multipolygon = annotation.to_shapely_annotation().multipolygon
    transformed_annotation_multipolygon = affine_transform(annotation_multipolygon, [a, b, d, e, x_off, y_off])
    return transformed_annotation_multipolygon

def prepare_propositions(pickle_path, orto_dir):
    assert pickle_path.exists(), f"Error: The file {pickle_path} does not exist."    
    logging.info(f"Processing: {pickle_path}")
    
    predictions = pickle.load(open(str(pickle_path), 'rb')) 
    if predictions:
        img_name = (pickle_path.with_suffix(".tif")).name
        orto_path = Path(orto_dir) / img_name
        
        images = []
        geometries = []
        for annotation in predictions:
            transformed_annotation = transform_annotation(annotation=annotation, orto_path=orto_path)  
            geometries.append(transformed_annotation)
            images.append(pickle_path.stem)  

        data = {
            'image': images,
            'score': [annotation.score.value for annotation in predictions],
            'category': [annotation.category.name for annotation in predictions],
            'geometry': geometries
        }
        return gpd.GeoDataFrame(data, crs='EPSG:3301')
    else:
        logging.info(f"No birds detected in orto: {pickle_path.stem}")

pickles_path = Path(r'G:\DT\DL\jcharyton\sahi_change_mask_type_repo\runs\predict\exp16\pickles').glob("*.pickle")

# for debugging
flight_propositions = []
for pickle_path in tqdm(pickles_path):
    flight_proposition = prepare_propositions(pickle_path, images_directory)
    flight_propositions.append(flight_proposition)

flight_gdf = gpd.GeoDataFrame(pd.concat(flight_propositions, ignore_index=True))

save_dir = Path(images_directory).parts[:-2] + ('predictions',) + Path(images_directory).parts[-1:]
output_dir = Path(*save_dir) / datetime.today().strftime('%Y-%m-%d')
output_dir.mkdir(parents=True, exist_ok=True)
filename = output_dir  / "propositions.gpkg"
flight_gdf.to_file(str(filename))