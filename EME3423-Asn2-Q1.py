# EME3423 Assignment 2 - Question 1
# Compare YOLO-608 vs YOLO-tiny detection speed (FPS)
# Run this script to compare both models

import cv2
import numpy as np
import time

# YOLO model paths
YOLO_MODELS = {
    'YOLO-608': {
        'weights': "c:/Users/Ryan/Downloads/yolov3 model/yolov3-608.weights",
        'config': "c:/Users/Ryan/Downloads/yolov3 model/yolov3-608.cfg",
    },
    'YOLO-tiny': {
        'weights': "c:/Users/Ryan/Downloads/yolov3 model/yolov3-tiny.weights",
        'config': "c:/Users/Ryan/Downloads/yolov3 model/yolov3-tiny.cfg",
    }
}

YOLO_NAMES = "c:/Users/Ryan/Downloads/yolov3 model/coco80.names"

# Load COCO class names
with open(YOLO_NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Setup camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("=" * 60)
print("YOLO FPS Comparison - YOLO-608 vs YOLO-tiny")
print("=" * 60)

# Benchmark function
def benchmark_model(model_name, weights, config, num_frames=100):
    print(f"\n[INFO] Loading {model_name}...")
    
    # Load YOLO network
    net = cv2.dnn.readNet(weights, config)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    # Set preferable backend and target (CPU for fair comparison)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    print(f"[INFO] {model_name} loaded. Running {num_frames} frames...")
    
    # Warmup
    ret, frame = cam.read()
    if ret:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        _ = net.forward(output_layers)
    
    # Benchmark
    start_time = time.time()
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cam.read()
        if not ret:
            break
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        frame_count += 1
        
        if frame_count % 20 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(f"  Progress: {frame_count}/{num_frames} frames, Current FPS: {current_fps:.2f}")
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    
    print(f"\n[RESULT] {model_name}:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    
    return avg_fps

# Run benchmarks
results = {}
for model_name, model_info in YOLO_MODELS.items():
    try:
        fps = benchmark_model(model_name, model_info['weights'], model_info['config'], num_frames=100)
        results[model_name] = fps
    except Exception as e:
        print(f"[ERROR] Failed to benchmark {model_name}: {e}")
        results[model_name] = None

# Summary
print("\n" + "=" * 60)
print("SUMMARY - FPS Comparison")
print("=" * 60)
for model, fps in results.items():
    if fps:
        print(f"  {model}: {fps:.2f} FPS")

if all(results.values()):
    faster = max(results, key=results.get)
    slower = min(results, key=results.get)
    speedup = results[faster] / results[slower]
    print(f"\n  {faster} is {speedup:.2f}x faster than {slower}")
    print(f"\n  Accuracy difference:")
    print(f"  - YOLO-608: Higher accuracy, detects smaller objects better")
    print(f"    More layers (106) and parameters, better at fine-grained detection")
    print(f"  - YOLO-tiny: Lower accuracy but much faster")
    print(f"    Fewer layers (24), fewer parameters, sacrifices accuracy for speed")

cam.release()
cv2.destroyAllWindows()
print("\n[INFO] Benchmark completed.")
