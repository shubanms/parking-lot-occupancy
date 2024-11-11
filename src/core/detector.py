import cv2

import numpy as np

from src.config import configs
from src.schemas.schemas import ParkingLotData

from src.service.services import create_grid, update_grid_with_iou, plot_parking_lot
from src.blob.blob_manager import blob_manager

def get_bounding_boxes(image_data, model_version):
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image_resized = cv2.cvtColor(cv2.resize(image, (640, 640)), cv2.COLOR_BGR2RGB)
    
    model = blob_manager.get_model(model_version)
    
    results = model(image_resized)
    
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()
    
    print(bounding_boxes)
    
    return bounding_boxes

def get_slot_occupancy(image_data, model_version):
    
    car_bounding_boxes = get_bounding_boxes(image_data, model_version)
    
    grid_pattern, slot_centers, slot_to_max_iou_bbox = create_grid(
        configs.section_lines, configs.section_capacities, car_bounding_boxes, configs.slot_width, configs.slot_length)

    grid_pattern, occupied_slots = update_grid_with_iou(grid_pattern, slot_to_max_iou_bbox)

    parking_lot_data = ParkingLotData(
        left=grid_pattern[-1][::-1],
        middle=grid_pattern[-2][::-1],
        right=grid_pattern[-3][::-1]
    )

    return parking_lot_data
