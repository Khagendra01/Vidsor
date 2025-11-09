import json
import numpy as np

# Load both JSON files
with open('kbc_yolo.json', 'r') as f:
    yolo_data = json.load(f)

with open('kbc_rfdetr.json', 'r') as f:
    rfdetr_data = json.load(f)

print("=== Bounding Box Comparison ===\n")

# Extract bboxes
yolo_bboxes = [det['bbox'] for det in yolo_data['detections']]
rfdetr_bboxes = []
for frame in rfdetr_data['frames']:
    for det in frame['detections']:
        rfdetr_bboxes.append(det['bbox'])

print(f"YOLO total bboxes: {len(yolo_bboxes)}")
print(f"RF-DETR total bboxes: {len(rfdetr_bboxes)}")

# Calculate bbox statistics
def bbox_stats(bboxes, name):
    bboxes = np.array(bboxes)
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    areas = widths * heights
    
    print(f"\n{name} BBox Statistics:")
    print(f"  Average width: {np.mean(widths):.2f} pixels")
    print(f"  Average height: {np.mean(heights):.2f} pixels")
    print(f"  Average area: {np.mean(areas):.2f} pixels²")
    print(f"  Min area: {np.min(areas):.2f} pixels²")
    print(f"  Max area: {np.max(areas):.2f} pixels²")
    print(f"  Average center X: {np.mean(bboxes[:, 0] + widths/2):.2f}")
    print(f"  Average center Y: {np.mean(bboxes[:, 1] + heights/2):.2f}")
    
    return bboxes, widths, heights, areas

yolo_bboxes_arr, yolo_w, yolo_h, yolo_areas = bbox_stats(yolo_bboxes, "YOLO")
rfdetr_bboxes_arr, rfdetr_w, rfdetr_h, rfdetr_areas = bbox_stats(rfdetr_bboxes, "RF-DETR")

# Compare similar detections (person class)
yolo_persons = [(det['bbox'], det['confidence']) for det in yolo_data['detections'] if det['class_name'] == 'person']
rfdetr_persons = []
for frame in rfdetr_data['frames']:
    for det in frame['detections']:
        if det['class_name'] == 'person':
            rfdetr_persons.append((det['bbox'], det['confidence']))

print(f"\n=== Person Detections ===")
print(f"YOLO persons: {len(yolo_persons)}")
print(f"RF-DETR persons: {len(rfdetr_persons)}")

# Calculate IoU for overlapping detections
def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Find matching detections using IoU
print(f"\n=== Matching Person Detections (IoU > 0.3) ===")
matched_pairs = []
iou_threshold = 0.3

for yolo_bbox, yolo_conf in yolo_persons[:50]:  # Check first 50 YOLO persons
    best_match = None
    best_iou = 0
    best_idx = -1
    
    for idx, (rfdetr_bbox, rfdetr_conf) in enumerate(rfdetr_persons):
        iou = calculate_iou(yolo_bbox, rfdetr_bbox)
        if iou > best_iou:
            best_iou = iou
            best_match = (rfdetr_bbox, rfdetr_conf)
            best_idx = idx
    
    if best_iou >= iou_threshold:
        matched_pairs.append((yolo_bbox, yolo_conf, best_match[0], best_match[1], best_iou))

print(f"Found {len(matched_pairs)} matching pairs (IoU >= {iou_threshold})")

if matched_pairs:
    ious = [pair[4] for pair in matched_pairs]
    print(f"Average IoU: {np.mean(ious):.3f}")
    print(f"Min IoU: {np.min(ious):.3f}")
    print(f"Max IoU: {np.max(ious):.3f}")
    
    # Show top 5 matches
    matched_pairs.sort(key=lambda x: x[4], reverse=True)
    print(f"\n=== Top 5 Matching Detections ===")
    for i, (yolo_bbox, yolo_conf, rfdetr_bbox, rfdetr_conf, iou) in enumerate(matched_pairs[:5]):
        yolo_bbox = np.array(yolo_bbox)
        rfdetr_bbox = np.array(rfdetr_bbox)
        
        diff_x1 = abs(yolo_bbox[0] - rfdetr_bbox[0])
        diff_y1 = abs(yolo_bbox[1] - rfdetr_bbox[1])
        diff_x2 = abs(yolo_bbox[2] - rfdetr_bbox[2])
        diff_y2 = abs(yolo_bbox[3] - rfdetr_bbox[3])
        
        yolo_w = yolo_bbox[2] - yolo_bbox[0]
        yolo_h = yolo_bbox[3] - yolo_bbox[1]
        rfdetr_w = rfdetr_bbox[2] - rfdetr_bbox[0]
        rfdetr_h = rfdetr_bbox[3] - rfdetr_bbox[1]
        
        print(f"\nMatch {i+1} (IoU: {iou:.3f}):")
        print(f"  YOLO:     [{yolo_bbox[0]:.1f}, {yolo_bbox[1]:.1f}, {yolo_bbox[2]:.1f}, {yolo_bbox[3]:.1f}] (w={yolo_w:.1f}, h={yolo_h:.1f}, conf={yolo_conf:.3f})")
        print(f"  RF-DETR:  [{rfdetr_bbox[0]:.1f}, {rfdetr_bbox[1]:.1f}, {rfdetr_bbox[2]:.1f}, {rfdetr_bbox[3]:.1f}] (w={rfdetr_w:.1f}, h={rfdetr_h:.1f}, conf={rfdetr_conf:.3f})")
        print(f"  Diff:     x1={diff_x1:.1f}, y1={diff_y1:.1f}, x2={diff_x2:.1f}, y2={diff_y2:.1f}")
        
        # Calculate center differences
        yolo_cx = (yolo_bbox[0] + yolo_bbox[2]) / 2
        yolo_cy = (yolo_bbox[1] + yolo_bbox[3]) / 2
        rfdetr_cx = (rfdetr_bbox[0] + rfdetr_bbox[2]) / 2
        rfdetr_cy = (rfdetr_bbox[1] + rfdetr_bbox[3]) / 2
        print(f"  Center diff: dx={abs(yolo_cx-rfdetr_cx):.1f}, dy={abs(yolo_cy-rfdetr_cy):.1f}")

# Compare overall differences
print(f"\n=== Overall BBox Size Comparison ===")
print(f"Average width difference: {abs(np.mean(yolo_w) - np.mean(rfdetr_w)):.2f} pixels")
print(f"Average height difference: {abs(np.mean(yolo_h) - np.mean(rfdetr_h)):.2f} pixels")
print(f"Average area difference: {abs(np.mean(yolo_areas) - np.mean(rfdetr_areas)):.2f} pixels²")
print(f"\nYOLO avg bbox size: {np.mean(yolo_w):.1f}x{np.mean(yolo_h):.1f}")
print(f"RF-DETR avg bbox size: {np.mean(rfdetr_w):.1f}x{np.mean(rfdetr_h):.1f}")

