import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
ESC_KEY = 27
MODEL_PATH = 'yoloe-11l-seg.pt'

# TRTM Visualization Parameters (Matches paper specs for display)
FLAT_PIXEL_VAL = 192       # Base gray level for flat cloth
PIXEL_PER_MM = 2.0         # 1cm (10mm) = 20 pixel units -> 1mm = 2 units
MANUAL_TABLE_DEPTH = None  # Set this (e.g., 850) if auto-detection is unstable

def find_best_config():
    """
    Auto-detects the best configuration for RealSense.
    """
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No RealSense device connected!")
        return None

    dev = ctx.devices[0]
    print(f"Device: {dev.get_info(rs.camera_info.name)}")
    
    config = rs.config()
    
    # Try to find a matching resolution (e.g. 640x480)
    target_w, target_h = 640, 480
    found_depth = False
    found_color = False

    # Check Depth
    for p in dev.first_depth_sensor().get_stream_profiles():
        if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
            vp = p.as_video_stream_profile()
            if vp.width() == target_w and vp.height() == target_h:
                config.enable_stream(rs.stream.depth, target_w, target_h, rs.format.z16, 30)
                found_depth = True
                break
    
    # Check Color
    for p in dev.first_color_sensor().get_stream_profiles():
        if p.stream_type() == rs.stream.color and p.format() == rs.format.bgr8:
            vp = p.as_video_stream_profile()
            if vp.width() == target_w and vp.height() == target_h:
                config.enable_stream(rs.stream.color, target_w, target_h, rs.format.bgr8, 30)
                found_color = True
                break

    print(f"Depth Stream Found: {found_depth}, Color Stream Found: {found_color}")

    if not found_depth: config.enable_stream(rs.stream.depth)
    if not found_color: config.enable_stream(rs.stream.color)

    return config

def apply_trtm_shading(depth_map, mask_bg):
    """ Applies TRTM paper shading (Gray table, White wrinkles) """
    valid_pixels = depth_map[depth_map > 0]
    
    if MANUAL_TABLE_DEPTH:
        table_depth = MANUAL_TABLE_DEPTH
    elif len(valid_pixels) > 0:
        table_depth = np.percentile(valid_pixels, 98)
    else:
        return np.zeros_like(depth_map, dtype=np.uint8)

    height_map = table_depth - depth_map
    processed = FLAT_PIXEL_VAL + (height_map * PIXEL_PER_MM)
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    processed[mask_bg] = 255

    print(f"Table Depth: {table_depth} mm")
    print(processed)
    
    return processed

def main():
    # 1. Initialize YOLO
    print(f"Loading YOLO model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        model.set_classes(["cloth", "towel", 'phone', 'teddy bear'])
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize Camera & Alignment
    pipeline = rs.pipeline()
    config = find_best_config()
    if not config: return

    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"Config failed, using defaults... ({e})")
        config = rs.config()
        config.enable_all_streams()
        pipeline.start(config)

    # --- ALIGNMENT SETUP ---
    # We align COLOR to DEPTH. 
    # This keeps Depth raw (Scientific Data) and warps Color to match it.
    align_to = rs.stream.depth
    align = rs.align(align_to)
    print("Alignment Initialized: Color -> Depth")

    # Setup Recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_realsense_aligned_{timestamp}"
    is_recording = False
    frame_count = 0

    print(f"\nReady. Saving to: {save_folder}")
    print("Controls: [S] Record Frame | [Q] Quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # --- PROCESS ALIGNMENT ---
            # This aligns the frames so they are 1:1 pixel matched
            aligned_frames = align.process(frames)
            
            # Get Aligned Frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # Validate
            if not depth_frame or not color_frame: continue

            # Convert to Numpy
            depth_raw_mm = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # No resizing needed! Dimensions now match automatically.
            height, width = depth_raw_mm.shape
            print(f"Frame Size: {width}x{height}")

            # --- YOLO SEGMENTATION ---
            results = model.predict(color_image, verbose=False, imgsz=width, conf=0.3)
            
            combined_mask = np.zeros((height, width), dtype=np.uint8)

            print('results none?', results[0].masks is None)

            if results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                for mask in masks_data:
                    # Resize mask to current aligned resolution
                    mask_resized = cv2.resize(mask, (width, height))
                    combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))
            
            binary_mask = combined_mask.astype(np.uint8)
            background_mask = (binary_mask == 0)

            # --- APPLY MASKS ---
            
            # 1. Mask Color
            masked_color = cv2.bitwise_and(color_image, color_image, mask=binary_mask)

            # 2. Mask Raw Depth (Saving)
            masked_raw_depth = cv2.bitwise_and(depth_raw_mm, depth_raw_mm, mask=binary_mask)

            # 3. TRTM Visualization (Display)
            trtm_display = apply_trtm_shading(depth_raw_mm, background_mask)
            trtm_display_bgr = cv2.cvtColor(trtm_display, cv2.COLOR_GRAY2BGR)

            # Stack side-by-side
            combined_display = np.hstack((masked_color, trtm_display_bgr))

            # --- RECORDING ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                is_recording = not is_recording
                if is_recording:
                    os.makedirs(save_folder, exist_ok=True)
                    print(f"[*] START RECORDING to {save_folder}")
                else:
                    print(f"[*] STOP RECORDING. Total: {frame_count}")

            if is_recording:
                depth_filename = os.path.join(save_folder, f"depth_{frame_count:05d}.png")
                color_filename = os.path.join(save_folder, f"color_{frame_count:05d}.png")

                # Save Data
                cv2.imwrite(depth_filename, masked_raw_depth) 
                cv2.imwrite(color_filename, masked_color)

                frame_count += 1
                cv2.circle(combined_display, (30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("RealSense Aligned TRTM", combined_display)

            if key == ord('q') or key == ESC_KEY:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()