# EME3423 Assignment 2 - Question 3
# YOLO Fruit Detection with Price Calculator
# Detects banana, apple, orange with confidence >= 80%
# Shows bounding box with name and confidence, total fruit count and price on upper right

import cv2
import numpy as np
import time

# ==================== CONFIGURATION ====================
# Fruit prices (in $)
FRUIT_PRICES = {
    'banana': 3,
    'apple': 5,
    'orange': 4
}

# COCO class IDs for fruits
FRUIT_CLASS_IDS = {47: 'banana', 48: 'apple', 50: 'orange'}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.80

# YOLO model paths
YOLO_WEIGHTS = "c:/Users/Ryan/Downloads/yolov3 model/yolov3-608.weights"
YOLO_CONFIG = "c:/Users/Ryan/Downloads/yolov3 model/yolov3-608.cfg"
YOLO_NAMES = "c:/Users/Ryan/Downloads/yolov3 model/coco80.names"

# ==================== LOAD YOLO ====================
print("[INFO] Loading YOLO model...")
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

# Load COCO class names
with open(YOLO_NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set preferable backend and target (CPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("[INFO] YOLO model loaded successfully!")

# ==================== CAMERA SETUP ====================
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# FPS calculation variables
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

# ==================== MAIN LOOP ====================
print("[INFO] Starting fruit detection... Press 'ESC' to exit.")

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to capture frame!")
        break

    # Calculate FPS
    fps_frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time >= 1.0:
        current_fps = fps_frame_count / elapsed_time
        fps_frame_count = 0
        fps_start_time = time.time()

    # Resize frame for processing
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    height, width, channels = frame.shape
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only process fruit classes with confidence >= threshold
            if class_id in FRUIT_CLASS_IDS and confidence >= CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.45)

    # Initialize counters
    fruit_counts = {'banana': 0, 'apple': 0, 'orange': 0}
    total_price = 0

    # Draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            fruit_name = FRUIT_CLASS_IDS[class_id]
            confidence = confidences[i]

            # Count fruits
            if fruit_name in fruit_counts:
                fruit_counts[fruit_name] += 1
                total_price += FRUIT_PRICES[fruit_name]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label background
            label_text = f"{fruit_name} {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, (x, y - label_h - 10), (x + label_w, y),
                (0, 255, 0), -1
            )

            # Draw label text (fruit name + confidence at top left of bounding box)
            cv2.putText(
                frame, label_text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

    # ========== Display total fruits and price at upper right corner ==========
    total_fruits = sum(fruit_counts.values())

    # Create info panel at upper right
    panel_x = width - 320
    panel_y = 10
    panel_w = 300
    panel_h = 120

    # Draw semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw border
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 0), 2)

    # Draw title
    cv2.putText(
        frame, "=== FRUIT CART ===",
        (panel_x + 15, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    # Draw fruit counts
    y_offset = panel_y + 50
    for fruit, count in fruit_counts.items():
        if count > 0:
            price_line = f"{fruit}: {count} x ${FRUIT_PRICES[fruit]} = ${count * FRUIT_PRICES[fruit]}"
            cv2.putText(
                frame, price_line,
                (panel_x + 15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 22

    # Draw total
    cv2.putText(
        frame, f"TOTAL FRUITS: {total_fruits}",
        (panel_x + 15, panel_y + panel_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
    )
    cv2.putText(
        frame, f"TOTAL PRICE: ${total_price}",
        (panel_x + 15, panel_y + panel_h - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
    )

    # ========== Display FPS at bottom left ==========
    cv2.putText(
        frame, f"FPS: {current_fps:.1f}",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    # Show frame
    cv2.imshow('YOLO Fruit Detection - Q3', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
print("[INFO] Fruit detection closed.")
