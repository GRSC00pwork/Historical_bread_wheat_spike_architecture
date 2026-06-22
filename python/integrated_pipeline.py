"""
Integrated Wheat Spike Phenotyping Pipeline
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.signal import savgol_filter
import torch
import argparse
import gc
import warnings
warnings.filterwarnings('ignore')
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class Config:
    SPIKE_MODEL_SIZE = 2560
    SPIKELET_MODEL_SIZE = 1920
    WORKING_SIZE = 2560
    SCALE_FACTOR = WORKING_SIZE / SPIKELET_MODEL_SIZE
    SPIKE_ORIENTATION = 'horizontal'
    SAVGOL_WINDOW_RATIO = 0.6
    SAVGOL_POLYORDER = 2
    TRIM_ENDPOINTS = 10
    USE_CALIBRATION = True
    REFERENCE_CIRCLE_DIAMETER_MM = 14.0
    CIRCLE_COLOR_LOWER = np.array([20, 100, 100])
    CIRCLE_COLOR_UPPER = np.array([35, 255, 255])
    SPIKE_NUM_CLASSES = 3
    SPIKE_ENCODER = 'resnet101'
    DECODER_ATTENTION = 'scse'
    USE_HEAD_ONLY = True
    SPIKELET_CONF_THRESHOLD = 0.5
    SPIKELET_IOU_THRESHOLD = 0.5
    DEBUG = True
# PREPROCESSING

def preprocess_image(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return padded, {'scale': scale, 'pad_x': pad_w, 'pad_y': pad_h,
                    'scaled_w': new_w, 'scaled_h': new_h, 'original_w': w, 'original_h': h}

def preprocess_for_model(image, target_size, normalize=True):
    processed, params = preprocess_image(image, target_size)
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    if normalize:
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    return tensor, params

# MODEL LOADING

def load_spike_model(model_path, config, device):
    import segmentation_models_pytorch as smp
    print(f"Loading spike model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    saved_config = checkpoint.get('config', {})
    encoder_name = saved_config.get('encoder_name', config.SPIKE_ENCODER)
    num_classes = saved_config.get('num_classes', config.SPIKE_NUM_CLASSES)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if config.DECODER_ATTENTION == 'auto':
        has_attention = any('attention' in key for key in state_dict.keys())
        decoder_attention_type = 'scse' if has_attention else None
    elif config.DECODER_ATTENTION == 'scse':
        decoder_attention_type = 'scse'
    else:
        decoder_attention_type = None
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                     in_channels=3, classes=num_classes, decoder_attention_type=decoder_attention_type)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"  ✅ Spike model loaded ({encoder_name}, attention={decoder_attention_type})")
    return model


def load_spikelet_model(model_path, config):
    from ultralytics import YOLO
    print(f"Loading spikelet model from: {model_path}")
    model = YOLO(model_path)
    print(f"  ✅ Spikelet model loaded")
    return model

# INFERENCE


def run_spike_inference(model, image, config, device):
    tensor, params = preprocess_for_model(image, config.SPIKE_MODEL_SIZE)
    tensor = tensor.unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    del tensor, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if config.USE_HEAD_ONLY:
        mask = (pred == 1).astype(np.uint8) * 255
    else:
        mask = ((pred == 1) | (pred == 2)).astype(np.uint8) * 255
    return mask, params


def run_spikelet_inference(model, image, config):
    processed, params = preprocess_image(image, config.SPIKELET_MODEL_SIZE)
    masks, classes, confidences = [], [], []
    try:
        results = model.predict(processed, conf=config.SPIKELET_CONF_THRESHOLD,
                                iou=config.SPIKELET_IOU_THRESHOLD, verbose=False,
                                device=0 if torch.cuda.is_available() else 'cpu')
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()
                    if mask_np.shape[0] != config.SPIKELET_MODEL_SIZE:
                        mask_np = cv2.resize(mask_np, (config.SPIKELET_MODEL_SIZE, config.SPIKELET_MODEL_SIZE),
                                             interpolation=cv2.INTER_NEAREST)
                    masks.append((mask_np > 0.5).astype(np.uint8) * 255)
                    if result.boxes is not None and i < len(result.boxes.cls):
                        classes.append(int(result.boxes.cls[i].cpu().numpy()))
                        confidences.append(float(result.boxes.conf[i].cpu().numpy()))
                    else:
                        classes.append(0)
                        confidences.append(1.0)
        del results
    except Exception as e:
        print(f"    Warning: Spikelet inference error: {e}")
    return masks, classes, confidences, params


def align_spikelet_to_working_size(mask, config):
    return cv2.resize(mask, (config.WORKING_SIZE, config.WORKING_SIZE), interpolation=cv2.INTER_NEAREST)

# MASK CLEANING
def clean_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <= 1:
        return mask
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
    return clean

# SKELETON & PATH EXTRACTION
def fast_skeletonize(mask):
    binary = (mask > 0).astype(np.uint8)
    try:
        return cv2.ximgproc.thinning(binary * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except (AttributeError, cv2.error):
        pass
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary).astype(np.uint8) * 255
    except ImportError:
        pass
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(binary)
    temp = binary.copy()
    for _ in range(100):
        eroded = cv2.erode(temp, kernel)
        opened = cv2.dilate(eroded, kernel)
        subset = temp - opened
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton * 255

def find_endpoints(skeleton):
    skel_binary = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_binary, -1, kernel)
    endpoints = np.where((skel_binary == 1) & (neighbor_count == 1))
    if len(endpoints[0]) == 0:
        return None
    return np.column_stack([endpoints[1], endpoints[0]])

def find_tip_and_base_points(skeleton, orientation='horizontal'):
    endpoints = find_endpoints(skeleton)
    if endpoints is None or len(endpoints) < 2:
        skel_points = np.column_stack(np.where(skeleton > 0))[:, [1, 0]]
        if len(skel_points) < 2:
            return None, None
        if orientation == 'horizontal':
            return skel_points[np.argmax(skel_points[:, 0])], skel_points[np.argmin(skel_points[:, 0])]
        else:
            return skel_points[np.argmin(skel_points[:, 1])], skel_points[np.argmax(skel_points[:, 1])]
    if orientation == 'horizontal':
        tip_idx = np.argmax(endpoints[:, 0])
        base_idx = np.argmin(endpoints[:, 0])
    elif orientation == 'vertical':
        tip_idx = np.argmax(endpoints[:, 1])
        base_idx = np.argmin(endpoints[:, 1])
    else:
        max_dist = 0
        tip_idx, base_idx = 0, 1
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                dist = np.linalg.norm(endpoints[i] - endpoints[j])
                if dist > max_dist:
                    max_dist = dist
                    tip_idx, base_idx = i, j
        if endpoints[tip_idx][0] > endpoints[base_idx][0]:
            tip_idx, base_idx = base_idx, tip_idx
    return endpoints[tip_idx], endpoints[base_idx]

def build_skeleton_graph(skeleton):
    skel_binary = (skeleton > 0).astype(np.uint8)
    points = np.column_stack(np.where(skel_binary > 0))[:, [1, 0]]
    if len(points) == 0:
        return None, None, None
    point_map = {tuple(p): i for i, p in enumerate(points)}
    graph = {i: [] for i in range(len(points))}
    for i, (x, y) in enumerate(points):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in point_map:
                    j = point_map[neighbor]
                    if j not in graph[i]:
                        graph[i].append(j)
    return graph, points, point_map

def dfs_path(graph, points, start_idx, end_idx):
    visited = set()
    stack = [(start_idx, [start_idx])]
    while stack:
        node, path = stack.pop()
        if node == end_idx:
            return np.array([points[i] for i in path])
        if node in visited:
            continue
        visited.add(node)
        end_point = points[end_idx]
        neighbors_sorted = sorted(graph[node], key=lambda n: np.linalg.norm(points[n] - end_point), reverse=True)
        for neighbor in neighbors_sorted:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None

def extract_main_path(skeleton, tip_point, base_point):
    result = build_skeleton_graph(skeleton)
    if result[0] is None:
        return None
    graph, points, point_map = result
    if len(points) > 10000:
        distances = np.linalg.norm(points - tip_point, axis=1)
        return points[np.argsort(distances)]
    tip_idx = point_map.get(tuple(tip_point))
    base_idx = point_map.get(tuple(base_point))
    if tip_idx is None or base_idx is None:
        tip_idx = np.argmin(np.linalg.norm(points - tip_point, axis=1))
        base_idx = np.argmin(np.linalg.norm(points - base_point, axis=1))
    path = dfs_path(graph, points, tip_idx, base_idx)
    if path is None:
        distances = np.linalg.norm(points - tip_point, axis=1)
        return points[np.argsort(distances)]
    return path

def smooth_path_savgol(path, window_ratio=0.4, polyorder=1):
    if path is None or len(path) < 5:
        return path
    x, y = path[:, 0].astype(float), path[:, 1].astype(float)
    window_length = max(5, int(len(path) * window_ratio))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(path))
    try:
        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)
        return np.column_stack([x_smooth, y_smooth])
    except:
        return path

def extract_spike_axis(spike_mask, config, debug=False):
    binary_mask = (spike_mask > 0).astype(np.uint8) * 255
    if np.sum(binary_mask) == 0:
        return None, None, None, None
    skeleton = fast_skeletonize(binary_mask)
    if debug:
        print(f"    Skeleton: {np.sum(skeleton > 0)} points")
    tip_point, base_point = find_tip_and_base_points(skeleton, config.SPIKE_ORIENTATION)
    if tip_point is None or base_point is None:
        return None, None, None, None
    path = extract_main_path(skeleton, tip_point, base_point)
    if path is None or len(path) < 5:
        return None, None, None, None
    smoothed_path = smooth_path_savgol(path, config.SAVGOL_WINDOW_RATIO, config.SAVGOL_POLYORDER)
    if config.TRIM_ENDPOINTS > 0 and len(smoothed_path) > 2 * config.TRIM_ENDPOINTS:
        smoothed_path = smoothed_path[config.TRIM_ENDPOINTS:-config.TRIM_ENDPOINTS]
    diffs = np.diff(smoothed_path, axis=0)
    arc_length = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))])
    return smoothed_path, arc_length, tip_point, base_point

# SPIKE TRAIT CALCULATION

def calculate_spike_length(path):
    if path is None or len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def calculate_spike_area_perimeter_roundness(mask, smooth_kernel_size=15):
    if mask is None or np.sum(mask) == 0:
        return 0, 0, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return area, perimeter, roundness

def calculate_spike_shape_metrics(mask, spike_axis, arc_length, smooth_kernel_size=15):
    if mask is None or np.sum(mask) == 0:
        return 1.0, 1.0, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0, 1.0, 0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    area_peri = area / perimeter if perimeter > 0 else 0

    axis_ratio = 1.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            major, minor = ellipse[1]
            if minor > major:
                major, minor = minor, major
            axis_ratio = major / minor if minor > 0 else 1.0
        except:
            pass

    area_ratio = 1.0
    if spike_axis is not None and arc_length is not None and len(spike_axis) > 2:
        mid_idx = len(spike_axis) // 2
        mid_point = spike_axis[mid_idx]
        if mid_idx < 2:
            tangent = spike_axis[mid_idx + 2] - spike_axis[mid_idx]
        elif mid_idx >= len(spike_axis) - 2:
            tangent = spike_axis[mid_idx] - spike_axis[mid_idx - 2]
        else:
            tangent = spike_axis[mid_idx + 2] - spike_axis[mid_idx - 2]
        if np.linalg.norm(tangent) > 0:
            tangent = tangent / np.linalg.norm(tangent)
            h, w = smoothed.shape
            y_c, x_c = np.ogrid[:h, :w]
            side = tangent[0] * (x_c - mid_point[0]) + tangent[1] * (y_c - mid_point[1])
            a1 = np.sum((side > 0) & (smoothed > 0))
            a2 = np.sum((side <= 0) & (smoothed > 0))
            if a2 > 0:
                area_ratio = a1 / a2

    return axis_ratio, area_ratio, area_peri

# SPIKE WIDTH
def calculate_spike_width_at_spikelet_positions(spike_mask, spike_axis, arc_length, spikelets_data, max_ray_distance=500):
    """Measure one-sided spike width at each fertile spikelet's centroid position."""
    if spike_axis is None or arc_length is None or len(spike_axis) < 10:
        return [], {}
    if spike_mask is None or np.sum(spike_mask) == 0:
        return [], {}
    if not spikelets_data:
        return [], {}

    clean_spike = clean_mask(spike_mask.copy())
    spikelet_widths = []

    for sp in spikelets_data:
        if sp.get('fertility') != 'fertile':
            continue
        centroid = sp.get('centroid')
        if centroid is None:
            continue

        distances_to_axis = np.linalg.norm(spike_axis - centroid, axis=1)
        closest_idx = np.argmin(distances_to_axis)
        axis_point = spike_axis[closest_idx]

        if closest_idx < 2:
            tangent = spike_axis[closest_idx + 2] - spike_axis[closest_idx]
        elif closest_idx >= len(spike_axis) - 2:
            tangent = spike_axis[closest_idx] - spike_axis[closest_idx - 2]
        else:
            tangent = spike_axis[closest_idx + 2] - spike_axis[closest_idx - 2]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            continue
        tangent = tangent / tangent_norm
        perpendicular = np.array([-tangent[1], tangent[0]])

        centroid_vector = centroid - axis_point
        dot_sign = np.dot(centroid_vector, perpendicular)
        if dot_sign >= 0:
            ray_direction = perpendicular
            side = 'right'
        else:
            ray_direction = -perpendicular
            side = 'left'

        half_width = 0
        for d in range(1, max_ray_distance):
            test_point = axis_point + ray_direction * d
            x, y = int(round(test_point[0])), int(round(test_point[1]))
            if x < 0 or x >= clean_spike.shape[1] or y < 0 or y >= clean_spike.shape[0]:
                half_width = d - 1
                break
            if clean_spike[y, x] == 0:
                half_width = d - 1
                break
            half_width = d

        if half_width < 1:
            continue

        spikelet_widths.append({
            'spikelet_id': sp.get('spikelet_id'),
            'position': sp.get('position', 0),
            'segment': sp.get('segment', 'middle'),
            'spike_width_px': half_width,
            'side': side,
        })

    # Aggregate
    seg_widths = {}
    for seg in ['top', 'middle', 'bottom']:
        for s in ['left', 'right']:
            vals = [sw['spike_width_px'] for sw in spikelet_widths if sw['segment'] == seg and sw['side'] == s]
            seg_widths[f'{seg}_{s}_list'] = vals
            seg_widths[f'{seg}_{s}_mean'] = np.mean(vals) if vals else 0
        seg_widths[f'{seg}_full'] = seg_widths[f'{seg}_left_mean'] + seg_widths[f'{seg}_right_mean']
        ml = seg_widths[f'{seg}_left_mean']
        mr = seg_widths[f'{seg}_right_mean']
        total = ml + mr
        seg_widths[f'{seg}_asymmetry'] = abs(ml - mr) / total if total > 0 else 0

    return spikelet_widths, seg_widths

# SPIKELET TRAIT CALCULATION

def calculate_spikelet_pca(mask):
    """Calculate centroid, major axis, length, AND minor axis width."""
    points = np.column_stack(np.where(mask > 0))
    if len(points) < 5:
        return None, None, 0, 0, None
    centroid = np.mean(points[:, [1, 0]], axis=0)
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    major_idx = np.argmax(eigenvalues)
    minor_idx = np.argmin(eigenvalues)
    major_axis = eigenvectors[:, major_idx].real[[1, 0]]

    # Length:extent along major axis
    proj_major = centered @ eigenvectors[:, major_idx].real
    length = proj_major.max() - proj_major.min()

    # Width:extent along minor axis
    proj_minor = centered @ eigenvectors[:, minor_idx].real
    width = proj_minor.max() - proj_minor.min()

    min_idx_pt, max_idx_pt = np.argmin(proj_major), np.argmax(proj_major)
    endpoints = np.array([points[min_idx_pt][[1, 0]], points[max_idx_pt][[1, 0]]])
    return centroid, major_axis, length, width, endpoints
def calculate_spikelet_basic_traits(mask):
    area_pixels = cv2.countNonZero(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area_pixels, 0, 0
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    area_contour = cv2.contourArea(largest)
    roundness = (4 * np.pi * area_contour) / (perimeter ** 2) if perimeter > 0 else 0
    return area_pixels, perimeter, roundness

def calculate_position_on_spike(centroid, spike_axis, arc_length):
    if spike_axis is None or arc_length is None or centroid is None:
        return 0.5
    distances = np.linalg.norm(spike_axis - centroid, axis=1)
    closest_idx = np.argmin(distances)
    return arc_length[closest_idx] / arc_length[-1]

def classify_segment(position):
    if position < 0.333:
        return 'top'
    elif position < 0.667:
        return 'middle'
    return 'bottom'

def calculate_spikelet_angle(major_axis, centroid, spike_axis):
    if major_axis is None or spike_axis is None or centroid is None:
        return 0
    distances = np.linalg.norm(spike_axis - centroid, axis=1)
    closest_idx = np.argmin(distances)
    if closest_idx == 0:
        tangent = spike_axis[1] - spike_axis[0]
    elif closest_idx == len(spike_axis) - 1:
        tangent = spike_axis[-1] - spike_axis[-2]
    else:
        tangent = spike_axis[closest_idx + 1] - spike_axis[closest_idx - 1]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
    major_axis_norm = major_axis / (np.linalg.norm(major_axis) + 1e-10)
    dot = np.clip(np.dot(tangent, major_axis_norm), -1, 1)
    return np.degrees(np.arccos(np.abs(dot)))

# CALIBRATION
def detect_calibration_circle(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, config.CIRCLE_COLOR_LOWER, config.CIRCLE_COLOR_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_circle, best_circ = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = (4 * np.pi * area) / (peri ** 2)
        if circ > best_circ and circ > 0.7:
            best_circ = circ
            best_circle = c
    if best_circle is None:
        return None
    (x, y), radius = cv2.minEnclosingCircle(best_circle)
    return (2 * radius) / config.REFERENCE_CIRCLE_DIAMETER_MM

# AGGREGATION HELPERS
def safe_ratio(a, b):
    return a / b if b > 0 else 0

def aggregate_spikelet_traits(spikelets_data, spike_length_px):

    seg_map = {'top': 'apex', 'middle': 'cen', 'bottom': 'base'}
    traits = ['angle', 'length_px', 'width_px', 'area_px', 'perimeter_px', 'roundness']

    # Group spikelets
    groups = {}
    for fert in ['fertile', 'sterile']:
        for seg in ['top', 'middle', 'bottom']:
            key = (fert, seg)
            groups[key] = [s for s in spikelets_data if s['fertility'] == fert and s['segment'] == seg]

    r = {}

    # Per-segment means for fertile and infertile
    for fert in ['fertile', 'sterile']:
        f_abbr = 'FSk' if fert == 'fertile' else 'FiSk'
        for seg in ['top', 'middle', 'bottom']:
            seg_abbr = seg_map[seg]
            sps = groups[(fert, seg)]

            # Skip central infertile (rarely has data)
            if fert == 'sterile' and seg == 'middle':
                continue

            for trait in traits:
                vals = [s[trait] for s in sps if s.get(trait) is not None and s.get(trait, 0) > 0]
                col = f'{seg_abbr}_{f_abbr}_{trait}'
                r[col] = np.mean(vals) if vals else 0

    # Per-segment gap for fertile spikelets
    for seg in ['top', 'middle', 'bottom']:
        seg_abbr = seg_map[seg]
        sps = sorted(groups[('fertile', seg)], key=lambda s: s.get('position', 0))
        gaps = []
        for i in range(len(sps) - 1):
            gap = sps[i + 1]['position'] - sps[i]['position']
            gaps.append(gap)
        # Store as normalized gap (will be converted to mm later)
        r[f'{seg_abbr}_FSk_gap_norm'] = np.mean(gaps) if gaps else 0

    # Counts
    for seg in ['top', 'middle', 'bottom']:
        seg_abbr = seg_map[seg]
        r[f'F_Sk_{seg_abbr}'] = len(groups[('fertile', seg)])
        r[f'FiSk_{seg_abbr}'] = len(groups[('sterile', seg)])

    r['TFSk'] = sum(1 for s in spikelets_data if s['fertility'] == 'fertile')
    r['TFiSk'] = sum(1 for s in spikelets_data if s['fertility'] == 'sterile')
    r['TSkS'] = len(spikelets_data)
    r['R_fertility'] = safe_ratio(r['TFSk'], r['TFiSk'])

    # Fertile spikelet count ratios
    r['R_FSk_cen-base'] = safe_ratio(r['F_Sk_cen'], r['F_Sk_base'])
    r['R_FSk_cen-apex'] = safe_ratio(r['F_Sk_cen'], r['F_Sk_apex'])
    r['R_FSk_apex-base'] = safe_ratio(r['F_Sk_apex'], r['F_Sk_base'])

    # Infertile spikelet count ratios
    r['R_FiSk_cen-base'] = safe_ratio(r['FiSk_cen'], r['FiSk_base'])
    r['R_FiSk_cen-apex'] = safe_ratio(r['FiSk_cen'], r['FiSk_apex'])
    r['R_FiSk_apex-base'] = safe_ratio(r['FiSk_apex'], r['FiSk_base'])

    # Fertile spikelet trait ratios (7 traits × 3 ratio types = 21)
    trait_abbr = {
        'angle': 'ang', 'length_px': 'L', 'width_px': 'W', 'area_px': 'A',
        'perimeter_px': 'P', 'roundness': 'R'
    }
    for trait, t_abbr in trait_abbr.items():
        a = r.get(f'apex_FSk_{trait}', 0)
        c = r.get(f'cen_FSk_{trait}', 0)
        b = r.get(f'base_FSk_{trait}', 0)
        r[f'R_FSk_{t_abbr}_cen-base'] = safe_ratio(c, b)
        r[f'R_FSk_{t_abbr}_cen-apex'] = safe_ratio(c, a)
        r[f'R_FSk_{t_abbr}_apex-base'] = safe_ratio(a, b)

    # Gap ratios
    ga = r.get('apex_FSk_gap_norm', 0)
    gc_val = r.get('cen_FSk_gap_norm', 0)
    gb = r.get('base_FSk_gap_norm', 0)
    r['R_FSk_gap_cen-base'] = safe_ratio(gc_val, gb)
    r['R_FSk_gap_cen-apex'] = safe_ratio(gc_val, ga)
    r['R_FSk_gap_apex-base'] = safe_ratio(ga, gb)

    # Derived per-segment ratios
    for seg in ['top', 'middle', 'bottom']:
        seg_abbr = seg_map[seg]
        l = r.get(f'{seg_abbr}_FSk_length_px', 0)
        w = r.get(f'{seg_abbr}_FSk_width_px', 0)
        a = r.get(f'{seg_abbr}_FSk_area_px', 0)
        p = r.get(f'{seg_abbr}_FSk_perimeter_px', 0)
        r[f'R_FSk_L-W_{seg_abbr}'] = safe_ratio(l, w)
        r[f'R_FSk_A-P_{seg_abbr}'] = safe_ratio(a, p)

    # Position of first/last fertile
    fertile_sorted = sorted([s for s in spikelets_data if s['fertility'] == 'fertile'],
                            key=lambda s: s.get('position', 0))
    r['first_FSk'] = fertile_sorted[0]['position'] if fertile_sorted else 0
    r['last_FSk'] = fertile_sorted[-1]['position'] if fertile_sorted else 0
    r['FZL_norm'] = r['last_FSk'] - r['first_FSk']

    # Spikelet density per segment (count / (spike_length / 3))
    seg_length = spike_length_px / 3.0 if spike_length_px > 0 else 1
    r['SkD_apex'] = r['F_Sk_apex'] / seg_length
    r['SkD_cen'] = r['F_Sk_cen'] / seg_length
    r['SkD_base'] = r['F_Sk_base'] / seg_length

    return r

# MAIN PIPELINE

def process_single_image(image_path, spike_model, spikelet_model, config, device, output_dir):
    image_name = Path(image_path).stem
    debug = config.DEBUG

    if debug:
        print(f"\n  Processing: {image_name}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"    ERROR: Could not load image")
        return None

    # SPIKE INFERENCE
    spike_mask, spike_params = run_spike_inference(spike_model, image, config, device)
    spike_mask = clean_mask(spike_mask)

    # SPIKELET INFERENCE
    spikelet_masks, spikelet_classes, spikelet_confs, spikelet_params = run_spikelet_inference(
        spikelet_model, image, config)

    if debug:
        print(f"    Detected {len(spikelet_masks)} spikelets")

    # SPIKE AXIS
    spike_axis, arc_length, tip_point, base_point = extract_spike_axis(spike_mask, config, debug)
    if spike_axis is None:
        del image, spike_mask, spikelet_masks
        clear_memory()
        return None

    # SPIKE TRAITS
    spike_length_px = calculate_spike_length(spike_axis)
    spike_area_px, spike_perimeter_px, spike_roundness = calculate_spike_area_perimeter_roundness(spike_mask)
    axis_ratio, area_ratio, area_peri = calculate_spike_shape_metrics(spike_mask, spike_axis, arc_length)

    # PROCESS SPIKELETS
    spikelets_data = []
    for i, (mask, cls, conf) in enumerate(zip(spikelet_masks, spikelet_classes, spikelet_confs)):
        mask_aligned = align_spikelet_to_working_size(mask, config)
        fertility = 'fertile' if cls == 0 else 'sterile'
        centroid, major_axis, length_px, width_px, endpoints = calculate_spikelet_pca(mask_aligned)
        if centroid is None:
            continue
        area_px, perimeter_px, roundness = calculate_spikelet_basic_traits(mask_aligned)
        position = calculate_position_on_spike(centroid, spike_axis, arc_length)
        segment = classify_segment(position)
        angle = calculate_spikelet_angle(major_axis, centroid, spike_axis)
        spikelets_data.append({
            'spikelet_id': i + 1, 'fertility': fertility, 'class_id': cls, 'confidence': conf,
            'centroid': centroid, 'major_axis': major_axis, 'position': position, 'segment': segment,
            'angle': angle, 'length_px': length_px, 'width_px': width_px,
            'area_px': area_px, 'perimeter_px': perimeter_px, 'roundness': roundness,
        })

    # SPIKE WIDTH
    spikelet_widths, seg_widths = calculate_spike_width_at_spikelet_positions(
        spike_mask, spike_axis, arc_length, spikelets_data)

    # Add spike width to spikelet records
    width_lookup = {sw['spikelet_id']: sw for sw in spikelet_widths}
    for sp in spikelets_data:
        wl = width_lookup.get(sp['spikelet_id'])
        if wl:
            sp['spike_width_at_position_px'] = wl['spike_width_px']
            sp['side'] = wl['side']
        else:
            sp['spike_width_at_position_px'] = None
            sp['side'] = None

    #  CALIBRATION
    pixels_per_mm = None
    if config.USE_CALIBRATION:
        pixels_per_mm = detect_calibration_circle(image, config)
        if pixels_per_mm and debug:
            print(f"    Calibration: {pixels_per_mm:.2f} px/mm")
    adjusted_ppm = pixels_per_mm * (config.WORKING_SIZE / max(image.shape[:2])) if pixels_per_mm else None

    # AGGREGATE SPIKELET TRAITS
    agg = aggregate_spikelet_traits(spikelets_data, spike_length_px)

    # SPIKE WIDTH AGGREGATES
    # Widest segment
    full_widths = {
        'apex': seg_widths.get('top_full', 0),
        'cen': seg_widths.get('middle_full', 0),
        'base': seg_widths.get('bottom_full', 0),
    }
    widest_seg = max(full_widths, key=full_widths.get) if any(v > 0 for v in full_widths.values()) else 'cen'
    widest_width = full_widths[widest_seg]

    # Store ppm for spikelet mm conversion
    for sp in spikelets_data:
        sp['adjusted_ppm'] = adjusted_ppm
        sp['image_name'] = image_name

    del image, spike_mask, spikelet_masks
    clear_memory()

    if debug:
        print(f"    ✓ {agg['TFSk']} fertile, {agg['TFiSk']} sterile")

    #RESULTS
    ppm = adjusted_ppm if adjusted_ppm else 1.0
    has_mm = adjusted_ppm is not None

    def to_mm(px_val):
        return px_val / ppm if has_mm else 0

    def to_mm2(px_val):
        return px_val / (ppm ** 2) if has_mm else 0

    def list_to_str(vals):
        return ', '.join([f'{v:.1f}' for v in vals]) if vals else ''

    def list_to_mm_str(vals):
        return ', '.join([f'{v/ppm:.2f}' for v in vals]) if vals and has_mm else ''

    return {
        'image_name': image_name,
        'has_mm': has_mm,
        'ppm': adjusted_ppm,

        #Spike px
        'SL_px': spike_length_px,
        'SA_px': spike_area_px,
        'SP_px': spike_perimeter_px,
        'SW_apex_l_px_list': list_to_str(seg_widths.get('top_left_list', [])),
        'SW_apex_r_px_list': list_to_str(seg_widths.get('top_right_list', [])),
        'SW_cen_l_px_list': list_to_str(seg_widths.get('middle_left_list', [])),
        'SW_cen_r_px_list': list_to_str(seg_widths.get('middle_right_list', [])),
        'SW_base_l_px_list': list_to_str(seg_widths.get('bottom_left_list', [])),
        'SW_base_r_px_list': list_to_str(seg_widths.get('bottom_right_list', [])),
        'mean_SW_apex_l_px': seg_widths.get('top_left_mean', 0),
        'mean_SW_apex_r_px': seg_widths.get('top_right_mean', 0),
        'mean_SW_cen_l_px': seg_widths.get('middle_left_mean', 0),
        'mean_SW_cen_r_px': seg_widths.get('middle_right_mean', 0),
        'mean_SW_base_l_px': seg_widths.get('bottom_left_mean', 0),
        'mean_SW_base_r_px': seg_widths.get('bottom_right_mean', 0),
        'SW_apex_px': seg_widths.get('top_full', 0),
        'SW_cen_px': seg_widths.get('middle_full', 0),
        'SW_base_px': seg_widths.get('bottom_full', 0),

        #Spike mm
        'SL_mm': to_mm(spike_length_px),
        'SA_mm2': to_mm2(spike_area_px),
        'SP_mm': to_mm(spike_perimeter_px),
        'SW_apex_l_mm_list': list_to_mm_str(seg_widths.get('top_left_list', [])),
        'SW_apex_r_mm_list': list_to_mm_str(seg_widths.get('top_right_list', [])),
        'SW_cen_l_mm_list': list_to_mm_str(seg_widths.get('middle_left_list', [])),
        'SW_cen_r_mm_list': list_to_mm_str(seg_widths.get('middle_right_list', [])),
        'SW_base_l_mm_list': list_to_mm_str(seg_widths.get('bottom_left_list', [])),
        'SW_base_r_mm_list': list_to_mm_str(seg_widths.get('bottom_right_list', [])),
        'mean_SW_apex_l_mm': to_mm(seg_widths.get('top_left_mean', 0)),
        'mean_SW_apex_r_mm': to_mm(seg_widths.get('top_right_mean', 0)),
        'mean_SW_cen_l_mm': to_mm(seg_widths.get('middle_left_mean', 0)),
        'mean_SW_cen_r_mm': to_mm(seg_widths.get('middle_right_mean', 0)),
        'mean_SW_base_l_mm': to_mm(seg_widths.get('bottom_left_mean', 0)),
        'mean_SW_base_r_mm': to_mm(seg_widths.get('bottom_right_mean', 0)),
        'SW_apex_mm': to_mm(seg_widths.get('top_full', 0)),
        'SW_cen_mm': to_mm(seg_widths.get('middle_full', 0)),
        'SW_base_mm': to_mm(seg_widths.get('bottom_full', 0)),

        # Spike unitless
        'SR': spike_roundness,
        'R_S_axis': axis_ratio,
        'R_S_area': area_ratio,
        'R_S_AP': area_peri,
        'R_SW_cen-apex': safe_ratio(seg_widths.get('middle_full', 0), seg_widths.get('top_full', 0)),
        'R_SW_cen-base': safe_ratio(seg_widths.get('middle_full', 0), seg_widths.get('bottom_full', 0)),
        'R_SW_apex-base': safe_ratio(seg_widths.get('top_full', 0), seg_widths.get('bottom_full', 0)),
        'R_SL-SW': safe_ratio(spike_length_px, widest_width),
        'SW_widest_seg': widest_seg,
        'SW_asym_apex': seg_widths.get('top_asymmetry', 0),
        'SW_asym_cen': seg_widths.get('middle_asymmetry', 0),
        'SW_asym_base': seg_widths.get('bottom_asymmetry', 0),

        #Counts
        **{k: v for k, v in agg.items() if k.startswith(('F_Sk', 'FiSk', 'TFSk', 'TFiSk', 'TSkS', 'R_fertility',
                                                          'R_FSk_cen', 'R_FSk_apex', 'R_FiSk'))},

        # Spikelet px 
        **{k: v for k, v in agg.items() if '_FSk_' in k and k.endswith('_px') and 'R_' not in k},
        **{k: v for k, v in agg.items() if '_FiSk_' in k and k.endswith('_px') and 'R_' not in k},

        # Spikelet mm 
        **{k.replace('_px', '_mm'): to_mm(v) for k, v in agg.items()
           if ('_FSk_length_px' in k or '_FSk_width_px' in k or '_FSk_perimeter_px' in k or
               '_FiSk_length_px' in k or '_FiSk_width_px' in k or '_FiSk_perimeter_px' in k)},
        **{k.replace('_px', '_mm2'): to_mm2(v) for k, v in agg.items()
           if ('_FSk_area_px' in k or '_FiSk_area_px' in k)},

        # Spikelet angles and roundness
        **{k: v for k, v in agg.items() if 'angle' in k and 'R_' not in k},
        **{k: v for k, v in agg.items() if 'roundness' in k and 'R_' not in k},

        #  Gap in mm 
        'apex_FSk_gap_mm': agg.get('apex_FSk_gap_norm', 0) * to_mm(spike_length_px) if has_mm else 0,
        'cen_FSk_gap_mm': agg.get('cen_FSk_gap_norm', 0) * to_mm(spike_length_px) if has_mm else 0,
        'base_FSk_gap_mm': agg.get('base_FSk_gap_norm', 0) * to_mm(spike_length_px) if has_mm else 0,

        # All ratios
        **{k: v for k, v in agg.items() if k.startswith('R_FSk_')},

        # Derived ratios
        **{k: v for k, v in agg.items() if k.startswith('R_FSk_L-W') or k.startswith('R_FSk_A-P')},

        # Position traits 
        'first_FSk': agg.get('first_FSk', 0),
        'last_FSk': agg.get('last_FSk', 0),
        'FZL_mm': agg.get('FZL_norm', 0) * to_mm(spike_length_px) if has_mm else 0,

        # Density 
        'SkD_apex': agg.get('SkD_apex', 0),
        'SkD_cen': agg.get('SkD_cen', 0),
        'SkD_base': agg.get('SkD_base', 0),

        #Individual spikelets
        'spikelets': spikelets_data,
    }

# SAVE — 5 SHEETS

def save_results(all_results, output_dir, filename='traits.xlsx'):
    excel_path = output_dir / filename

    # Collect all individual spikelets
    all_spikelets = []
    for res in all_results:
        for sp in res.get('spikelets', []):
            all_spikelets.append(sp)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        #Sheet 1: Spike (px)
        spike_px_cols = ['image_name', 'SL_px', 'SA_px', 'SP_px',
                         'SW_apex_l_px_list', 'SW_apex_r_px_list',
                         'SW_cen_l_px_list', 'SW_cen_r_px_list',
                         'SW_base_l_px_list', 'SW_base_r_px_list',
                         'mean_SW_apex_l_px', 'mean_SW_apex_r_px',
                         'mean_SW_cen_l_px', 'mean_SW_cen_r_px',
                         'mean_SW_base_l_px', 'mean_SW_base_r_px',
                         'SW_apex_px', 'SW_cen_px', 'SW_base_px']
        df1 = pd.DataFrame([{k: r.get(k) for k in spike_px_cols} for r in all_results])
        df1.to_excel(writer, sheet_name='Spike (px)', index=False)

        #Sheet 2: Spike (mm)
        spike_mm_cols = ['image_name', 'SL_mm', 'SA_mm2', 'SP_mm', 'SR', 'R_S_axis', 'R_S_area', 'R_S_AP',
                         'SW_apex_l_mm_list', 'SW_apex_r_mm_list',
                         'SW_cen_l_mm_list', 'SW_cen_r_mm_list',
                         'SW_base_l_mm_list', 'SW_base_r_mm_list',
                         'mean_SW_apex_l_mm', 'mean_SW_apex_r_mm',
                         'mean_SW_cen_l_mm', 'mean_SW_cen_r_mm',
                         'mean_SW_base_l_mm', 'mean_SW_base_r_mm',
                         'SW_apex_mm', 'SW_cen_mm', 'SW_base_mm',
                         'R_SL-SW', 'SW_widest_seg',
                         'R_SW_cen-apex', 'R_SW_cen-base', 'R_SW_apex-base',
                         'SW_asym_apex', 'SW_asym_cen', 'SW_asym_base',
                         'F_Sk_apex', 'F_Sk_cen', 'F_Sk_base',
                         'FiSk_apex', 'FiSk_cen', 'FiSk_base',
                         'TFSk', 'TFiSk', 'TSkS', 'R_fertility',
                         'R_FSk_cen-base', 'R_FSk_cen-apex', 'R_FSk_apex-base',
                         'R_FiSk_cen-base', 'R_FiSk_cen-apex', 'R_FiSk_apex-base']
        df2 = pd.DataFrame([{k: r.get(k) for k in spike_mm_cols} for r in all_results])
        df2.to_excel(writer, sheet_name='Spike (mm)', index=False)

        #Sheet 3: Spikelet (px)
        spikelet_px_cols = ['image_name']
        for seg in ['apex', 'cen', 'base']:
            for fert in ['FSk', 'FiSk']:
                if fert == 'FiSk' and seg == 'cen':
                    continue
                for trait in ['length_px', 'width_px', 'area_px', 'perimeter_px']:
                    spikelet_px_cols.append(f'{seg}_{fert}_{trait}')
        df3 = pd.DataFrame([{k: r.get(k) for k in spikelet_px_cols} for r in all_results])
        df3.to_excel(writer, sheet_name='Spikelet (px)', index=False)

        #Sheet 4: Spikelet (mm)
        spikelet_mm_cols = ['image_name']
        # Fertile traits per segment (mm)
        for seg in ['apex', 'cen', 'base']:
            spikelet_mm_cols.append(f'{seg}_FSk_angle')
            for trait in ['length_mm', 'width_mm', 'area_mm2', 'perimeter_mm']:
                spikelet_mm_cols.append(f'{seg}_FSk_{trait}')
            spikelet_mm_cols.append(f'{seg}_FSk_roundness')
        # Fertile gap (mm)
        for seg in ['apex', 'cen', 'base']:
            spikelet_mm_cols.append(f'{seg}_FSk_gap_mm')
        # Infertile traits (apical and basal only)
        for seg in ['apex', 'base']:
            spikelet_mm_cols.append(f'{seg}_FiSk_angle')
            for trait in ['length_mm', 'width_mm', 'area_mm2', 'perimeter_mm']:
                spikelet_mm_cols.append(f'{seg}_FiSk_{trait}')
            spikelet_mm_cols.append(f'{seg}_FiSk_roundness')
        # All ratios
        for trait_abbr in ['ang', 'L', 'W', 'A', 'P', 'R', 'gap']:
            for ratio in ['cen-base', 'cen-apex', 'apex-base']:
                spikelet_mm_cols.append(f'R_FSk_{trait_abbr}_{ratio}')
        # Derived
        for seg in ['apex', 'cen', 'base']:
            spikelet_mm_cols.append(f'R_FSk_L-W_{seg}')
            spikelet_mm_cols.append(f'R_FSk_A-P_{seg}')
        # Position, zone, density
        spikelet_mm_cols.extend(['first_FSk', 'last_FSk', 'FZL_mm',
                                 'SkD_apex', 'SkD_cen', 'SkD_base'])
        df4 = pd.DataFrame([{k: r.get(k) for k in spikelet_mm_cols} for r in all_results])
        df4.to_excel(writer, sheet_name='Spikelet (mm)', index=False)

        #Sheet 5: Individual spikelets (mm)
        if all_spikelets:
            records = []
            for sp in all_spikelets:
                ppm_val = sp.get('adjusted_ppm')
                if not ppm_val or ppm_val == 'N/A':
                    ppm_val = None
                rec = {
                    'image_name': sp.get('image_name'),
                    'spikelet_id': sp.get('spikelet_id'),
                    'fertility': sp.get('fertility'),
                    'segment': sp.get('segment'),
                    'position': sp.get('position'),
                    'angle': sp.get('angle'),
                    'length_mm': sp['length_px'] / ppm_val if ppm_val and sp.get('length_px') else 0,
                    'width_mm': sp['width_px'] / ppm_val if ppm_val and sp.get('width_px') else 0,
                    'area_mm2': sp['area_px'] / (ppm_val**2) if ppm_val and sp.get('area_px') else 0,
                    'perimeter_mm': sp['perimeter_px'] / ppm_val if ppm_val and sp.get('perimeter_px') else 0,
                    'roundness': sp.get('roundness'),
                    'spike_width_at_position_mm': sp['spike_width_at_position_px'] / ppm_val if ppm_val and sp.get('spike_width_at_position_px') else 0,
                    'side': sp.get('side'),
                }
                records.append(rec)
            df5 = pd.DataFrame(records)
            df5.to_excel(writer, sheet_name='Individual spikelets', index=False)

    print(f"    Saved: {excel_path}")

# MAIN

def main():
    parser = argparse.ArgumentParser(description='Wheat Spike Phenotyping Pipeline v6')
    parser.add_argument('--spike-model', required=True)
    parser.add_argument('--spikelet-model', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--spike-size', type=int, default=2560)
    parser.add_argument('--spikelet-size', type=int, default=1920)
    parser.add_argument('--orientation', choices=['horizontal', 'vertical', 'auto'], default='horizontal')
    parser.add_argument('--no-calibration', action='store_true')
    parser.add_argument('--circle-diameter', type=float, default=14.0)
    parser.add_argument('--head-only', action='store_true', default=True)
    parser.add_argument('--decoder-attention', choices=['auto', 'scse', 'none'], default='auto')
    parser.add_argument('--conf-threshold', type=float, default=0.65)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()

    config = Config()
    config.SPIKE_MODEL_SIZE = args.spike_size
    config.SPIKELET_MODEL_SIZE = args.spikelet_size
    config.WORKING_SIZE = args.spike_size
    config.SCALE_FACTOR = args.spike_size / args.spikelet_size
    config.SPIKE_ORIENTATION = args.orientation
    config.USE_CALIBRATION = not args.no_calibration
    config.REFERENCE_CIRCLE_DIAMETER_MM = args.circle_diameter
    config.USE_HEAD_ONLY = args.head_only
    config.DECODER_ATTENTION = args.decoder_attention
    config.SPIKELET_CONF_THRESHOLD = args.conf_threshold
    config.DEBUG = args.debug

    spike_model_path = Path(args.spike_model)
    spikelet_model_path = Path(args.spikelet_model)
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    for p, name in [(spike_model_path, 'Spike model'), (spikelet_model_path, 'Spikelet model'), (images_dir, 'Images')]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("WHEAT SPIKE PHENOTYPING PIPELINE v6")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Spike: {spike_model_path} | Spikelet: {spikelet_model_path}")
    print(f"Images: {images_dir} | Output: {output_dir}")
    print("=" * 70)

    spike_model = load_spike_model(spike_model_path, config, device)
    spikelet_model = load_spikelet_model(spikelet_model_path, config)

    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        return

    print(f"\nProcessing {len(image_files)} images...\n")

    all_results = []
    failed = []
    batch_number = 0
    total_processed = 0

    for idx, image_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] {image_path.name}")
        try:
            result = process_single_image(image_path, spike_model, spikelet_model, config, device, output_dir)
            if result is not None:
                all_results.append(result)
            else:
                failed.append(image_path.name)
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append(image_path.name)
        clear_memory()

        if (idx + 1) % args.batch_size == 0 and all_results:
            batch_number += 1
            save_results(all_results, output_dir, f'batch_{batch_number:03d}.xlsx')
            total_processed += len(all_results)
            all_results = []
            gc.collect()

    if all_results:
        batch_number += 1
        save_results(all_results, output_dir, f'batch_{batch_number:03d}.xlsx')
        total_processed += len(all_results)

    print("\n" + "=" * 70)
    print(f"COMPLETE: {total_processed} processed, {len(failed)} failed, {batch_number} batch(es)")
    if failed:
        for f in failed[:10]:
            print(f"  - {f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()