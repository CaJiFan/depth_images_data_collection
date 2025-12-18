import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
ESC_KEY = 27
MODEL_PATH = 'yolov8x-worldv2'

def find_best_config():
    """
    Auto-detects the best configuration for RealSense (handles USB 2.0 vs 3.0).
    """
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No RealSense device connected!")
        return None

    dev = ctx.devices[0]
    print(f"Device: {dev.get_info(rs.camera_info.name)}")
    
    config = rs.config()
    
    # 1. Try to find a matching resolution for both sensors (e.g. 640x480)
    # This helps minimize alignment errors later
    found_depth = False
    found_color = False
    target_w, target_h = 424, 240

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

    # Fallbacks if exact 640x480 isn't available (e.g. USB 2.0 limits)
    if not found_depth:
        print("Defaulting Depth stream (Specific resolution not found)")
        config.enable_stream(rs.stream.depth)
    if not found_color:
        print("Defaulting Color stream (Specific resolution not found)")
        config.enable_stream(rs.stream.color)

    return config

def main():
    # 1. Initialize YOLO
    print(f"Loading YOLO model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    model.set_classes(["cloth", "towel"])
    print("Model loaded.")

    # 2. Initialize Camera
    pipeline = rs.pipeline()
    config = find_best_config()
    
    if not config: return

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Config failed ({e}), attempting 'enable_all_streams' fallback...")
        config = rs.config()
        config.enable_all_streams()
        profile = pipeline.start(config)

    # Optional: Align Object (Aligns Depth to Color or vice versa)
    # Using 'align' ensures the RGB mask fits the Depth image perfectly physically.
    # However, aligning Depth to Color changes the Depth resolution to match Color.
    # For now, we will perform manual resizing to keep Depth RAW (Native Resolution).
    # align = rs.align(rs.stream.color) 

    # Setup Recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_realsense_yolo_{timestamp}"
    is_recording = False
    frame_count = 0

    print(f"\nReady. Saving to: {save_folder}")
    print("Controls: [S] Record Frame | [Q] Quit")

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Get Frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            # Convert to Numpy
            depth_raw_mm = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            height_d, width_d = depth_raw_mm.shape
            height_c, width_c = color_image.shape[:2]

            # --- YOLO SEGMENTATION ---
            # Run inference on Color
            results = model.predict(color_image, verbose=False, imgsz=424, conf=0.3)
            
            # Create Blank Mask
            combined_mask = np.zeros((height_c, width_c), dtype=np.uint8)

            if results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                for mask in masks_data:
                    mask_resized = cv2.resize(mask, (width_c, height_c))
                    combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))
            
            binary_mask_c = combined_mask.astype(np.uint8)

            # --- APPLY MASKS ---
            
            # 1. Mask Color
            mask_c_3ch = cv2.merge([binary_mask_c, binary_mask_c, binary_mask_c])
            masked_color = cv2.bitwise_and(color_image, color_image, mask=binary_mask_c)

            # 2. Mask Depth
            # Resize the Color Mask to fit Depth dimensions
            # (Essential because RealSense Depth often differs from RGB size)
            binary_mask_d = cv2.resize(binary_mask_c, (width_d, height_d), interpolation=cv2.INTER_NEAREST)
            masked_depth = cv2.bitwise_and(depth_raw_mm, depth_raw_mm, mask=binary_mask_d)

            # --- VISUALIZATION ---
            # Depth Visualization (Grayscale)
            depth_display = cv2.convertScaleAbs(masked_depth, alpha=0.03)
            depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

            # Resize Color to match Depth Height for clean side-by-side
            if height_c != height_d:
                aspect_ratio = width_c / height_c
                new_w = int(height_d * aspect_ratio)
                color_display = cv2.resize(masked_color, (new_w, height_d))
            else:
                color_display = masked_color

            combined_display = np.hstack((color_display, depth_display))

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
                # Save MASKED raw data
                depth_filename = os.path.join(save_folder, f"depth_{frame_count:05d}.png")
                color_filename = os.path.join(save_folder, f"color_{frame_count:05d}.png")

                cv2.imwrite(depth_filename, masked_depth) # Raw 16-bit
                cv2.imwrite(color_filename, masked_color)

                frame_count += 1
                cv2.circle(combined_display, (30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("RealSense YOLO Collector", combined_display)

            if key == ord('q') or key == ESC_KEY:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()