import os
import pickle
import time
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

import pandas as pd
import geopandas as gpd

from osgeo import gdal
from affine import Affine
from shapely.affinity import affine_transform

from sahi.predict import predict


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

        
def get_subdirs(root_dir):
    subdirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            subdirs.append(entry.path)
    return subdirs
  
def make_propositions(model_path, source_dir): 
    export_dict = predict(
            model_type='yolov8',
            model_path=model_path,
            model_device='cuda:0',
            model_confidence_threshold=0.3,
            source=source_dir,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            verbose=2,
            novisual=True,
            export_pickle=True,
            return_dict=True,
            postprocess_type="GREEDYNMM"
    )
    return export_dict

def main(model_path: str, source_dir: str):
    logging.info("Started generating predictions!")
    
    orto_dirs = get_subdirs(source_dir)
    
    for orto_dir in orto_dirs:
        clock_all = time.time()
        
        result = make_propositions(model_path, orto_dir)
        logging.info(f"Preparing predicion gdf for the following flight line: {orto_dir}")
        
        pickles_path = Path(result["export_dir"]/"pickles").glob("*.pickle")

        # for production        
        # flight_propositions = Parallel(n_jobs=-1)(
        #     delayed(prepare_propositions)(pickle, orto_dir) for pickle in tqdm(pickles_path)
        # )
        
        # for debugging
        flight_propositions = []
        for pickle in tqdm(pickles_path):
            flight_proposition = prepare_propositions(pickle, orto_dir)
            flight_propositions.append(flight_proposition)
        
        flight_gdf = gpd.GeoDataFrame(pd.concat(flight_propositions, ignore_index=True))
        
        save_dir = Path(orto_dir).parts[:-2] + ('predictions',) + Path(orto_dir).parts[-1:]
        output_dir = Path(*save_dir) / datetime.today().strftime('%Y-%m-%d')
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir  / "propositions.gpkg"
        flight_gdf.to_file(str(filename))
        
        logging.info("Time = %.3f sec", time.time() - clock_all)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model_path", type=str, default="v187-b112-S.pt", help="Model path to use for inference.")
    parser.add_argument("--source_dir", type=str, default=r"F:\DT\Estonia_Birds_2023\Collection_1\inference_toy_small", help="Source directory containing raw orthos for inference.")
    args = parser.parse_args()
    
    main(args.yolo_model_path, args.source_dir)
    