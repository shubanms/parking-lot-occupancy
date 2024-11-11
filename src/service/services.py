import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(slot_box, bbox):
    x1, y1, x2, y2 = slot_box
    bx1, by1, bx2, by2 = bbox

    inter_x1 = max(x1, bx1)
    inter_y1 = max(y1, by1)
    inter_x2 = min(x2, bx2)
    inter_y2 = min(y2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    slot_area = (x2 - x1) * (y2 - y1)
    bbox_area = (bx2 - bx1) * (by2 - by1)

    union_area = slot_area + bbox_area - intersection_area
    return intersection_area / union_area


def rotate_point(x, y, cx, cy, cos_angle, sin_angle):
    dx, dy = x - cx, y - cy
    return cx + cos_angle * dx - sin_angle * dy, cy + sin_angle * dx + cos_angle * dy


def rotate_bbox(bbox, angle_deg):
    angle_rad = math.radians(angle_deg)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)

    rotated_bbox = [
        *rotate_point(bbox[0], bbox[1], cx, cy,
                      cos_angle, sin_angle),  # (x1, y1)
        *rotate_point(bbox[2], bbox[1], cx, cy,
                      cos_angle, sin_angle),  # (x2, y1)
        *rotate_point(bbox[2], bbox[3], cx, cy,
                      cos_angle, sin_angle),  # (x2, y2)
        # (x1, y2)
        *rotate_point(bbox[0], bbox[3], cx, cy, cos_angle, sin_angle)
    ]
    return rotated_bbox


def compute_max_iou_for_slot(slot_box, car_bounding_boxes):
    max_iou = 0
    best_bbox = None
    for bbox_idx, bbox in enumerate(car_bounding_boxes):
        iou = calculate_iou(slot_box, bbox)
        if iou > max_iou:
            max_iou = iou
            best_bbox = bbox_idx
    return max_iou, best_bbox


def create_grid(section_lines, section_capacities, car_bounding_boxes, slot_width, slot_length, car_angles=None):
    grid_pattern = []
    slot_centers = {}
    slot_to_max_iou_bbox = {}

    adjusted_slot_width = slot_width * 1.1
    adjusted_slot_length = slot_length * 1.1

    if car_angles:
        car_bounding_boxes = [rotate_bbox(bbox, angle) for bbox, angle in zip(
            car_bounding_boxes, car_angles)]

    for section_idx, (start, end) in enumerate(section_lines):
        num_slots = section_capacities[section_idx]
        section_row = [-1] if section_idx == 1 else []

        for slot_num in range(num_slots):
            x = start[0] + (end[0] - start[0]) * (slot_num / num_slots)
            y = start[1] + (end[1] - start[1]) * (slot_num / num_slots)
            slot_box = [x - adjusted_slot_width / 2, y - adjusted_slot_length / 2,
                        x + adjusted_slot_width / 2, y + adjusted_slot_length / 2]

            slot_centers[(section_idx, slot_num)] = (x, y)

            max_iou, best_bbox = compute_max_iou_for_slot(
                slot_box, car_bounding_boxes)

            if max_iou > 0.3:
                if best_bbox not in slot_to_max_iou_bbox:
                    slot_to_max_iou_bbox[best_bbox] = (
                        section_idx, slot_num, max_iou)
                else:
                    if slot_to_max_iou_bbox[best_bbox][2] < max_iou:
                        slot_to_max_iou_bbox[best_bbox] = (
                            section_idx, slot_num, max_iou)

            section_row.append(0)

        if section_idx == 1:
            section_row.append(-1)

        grid_pattern.append(section_row)

    return grid_pattern, slot_centers, slot_to_max_iou_bbox

def update_grid_with_iou(grid_pattern, slot_to_max_iou_bbox):
    occupied_slots = set()
    for best_bbox, (section_idx, slot_num, iou) in slot_to_max_iou_bbox.items():
        grid_pattern[section_idx][slot_num] = 1
        occupied_slots.add((section_idx, slot_num))
    return grid_pattern, occupied_slots
