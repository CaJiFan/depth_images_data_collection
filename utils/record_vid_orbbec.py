import cv2
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO
from pyorbbecsdk import *

# --- CONFIGURATION ---
ESC_KEY = 27
MODEL_PATH = 'yoloe-11l-seg.pt' 

# TRTM Visualization Params
TARGET_CANVAS_SIZE = 720   # Final Output Size
CLOTH_ROI_SIZE = 480       # Crop Size
FLAT_PIXEL_VAL = 192       
PIXEL_PER_MM = 2.0         
MANUAL_TABLE_DEPTH = None  

def frame_to_bgr_image(frame: VideoFrame):
    """ Robustly converts an Orbbec VideoFrame to a standard OpenCV BGR image. """
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    try:
        if color_format == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            return np.resize(data, (height, width, 3))
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    except Exception:
        return None
    return None

def manual_align_color_to_depth(color_img, depth_w, depth_h):
    """
    Manually simulates C2D Alignment.
    1. Calculates Depth Aspect Ratio.
    2. Center-Crops Color image to match that ratio.
    3. Resizes Color to match Depth resolution.
    """
    h_c, w_c = color_img.shape[:2]
    
    # Target Aspect Ratio (Depth Sensor)
    target_aspect = depth_w / depth_h
    current_aspect = w_c / h_c
    
    crop_x = 0
    crop_y = 0
    crop_w = w_c
    crop_h = h_c

    # Determine Crop Region
    if current_aspect > target_aspect:
        # Color is "wider" than Depth (e.g., 16:9 vs 4:3) -> Crop sides
        new_w = int(h_c * target_aspect)
        crop_x = (w_c - new_w) // 2
        crop_w = new_w
    else:
        # Color is "taller" -> Crop top/bottom
        new_h = int(w_c / target_aspect)
        crop_y = (h_c - new_h) // 2
        crop_h = new_h

    # Perform Crop
    color_cropped = color_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    # Perform Resize to exact Depth dimensions
    color_aligned = cv2.resize(color_cropped, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
    
    return color_aligned

def center_crop_and_pad(img, target_roi=480, target_canvas=720, bg_color=0):
    """
    TRTM Formatting: 
    1. Crops center 480x480. 
    2. Pads to 720x720 canvas.
    """
    h, w = img.shape[:2]
    is_rgb = len(img.shape) == 3
    
    # 1. Define Crop Coordinates (Center)
    start_x = max(0, (w - target_roi) // 2)
    start_y = max(0, (h - target_roi) // 2)
    end_x = start_x + target_roi
    end_y = start_y + target_roi
    
    # Extract ROI
    cropped = img[start_y:end_y, start_x:end_x]
    
    # 2. Prepare Canvas
    if is_rgb:
        canvas = np.full((target_canvas, target_canvas, 3), bg_color, dtype=np.uint8)
    else:
        canvas = np.full((target_canvas, target_canvas), bg_color, dtype=img.dtype) # Preserve uint16 for raw depth
        
    # 3. Place ROI in Center of Canvas
    offset_x = (target_canvas - target_roi) // 2
    offset_y = (target_canvas - target_roi) // 2
    
    # Handle edge case if input was smaller than ROI
    c_h, c_w = cropped.shape[:2]
    
    if is_rgb:
        canvas[offset_y:offset_y+c_h, offset_x:offset_x+c_w, :] = cropped
    else:
        canvas[offset_y:offset_y+c_h, offset_x:offset_x+c_w] = cropped
        
    return canvas

def apply_trtm_shading(depth_map, mask_bg):
    """ Applies TRTM paper shading (White background, Gray cloth). """
    valid_pixels = depth_map[depth_map > 0]
    
    if MANUAL_TABLE_DEPTH:
        table_depth = MANUAL_TABLE_DEPTH
    elif len(valid_pixels) > 0:
        table_depth = np.percentile(valid_pixels, 98)
    else:
        return np.full_like(depth_map, 255, dtype=np.uint8), 0

    height_map = table_depth - depth_map
    processed = FLAT_PIXEL_VAL + (height_map * PIXEL_PER_MM)
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    # Set background to White (255)
    processed[mask_bg] = 255 
    
    return processed, table_depth

def main():
    # 1. Initialize YOLO
    print(f"Loading YOLO model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        model.set_classes(["cloth", "towel"])
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize Camera
    config = Config()
    pipeline = Pipeline()

    try:
        # Depth Profile (Default is usually 640x576)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        
        # Color Profile (Default is usually 1920x1080 MJPG)
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        
        # Disable SDK Align (We do it manually)
        try:
            config.set_align_mode(OBAlignMode.DISABLE)
        except:
            pass # V2 SDK sometimes defaults to disabled
            
        pipeline.start(config)
        print(f"Camera Started.")
        print(f"Depth Stream: {depth_profile.get_width()}x{depth_profile.get_height()}")
        print(f"Color Stream: {color_profile.get_width()}x{color_profile.get_height()}")

    except Exception as e:
        print(f"Camera Init Failed: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_orbbec_snapshots_{timestamp}"
    frame_count = 0
    show_flash = 0

    print(f"\nReady. Snapshots to: {save_folder}")
    print("Controls: [S] Snapshot | [Q] Quit")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None: continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            # --- PROCESS DEPTH (Master) ---
            width_d = depth_frame.get_width()
            height_d = depth_frame.get_height()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_raw_mm = depth_data.reshape((height_d, width_d))
            
            scale = depth_frame.get_depth_scale()
            depth_raw_mm = (depth_raw_mm * scale).astype(np.uint16)

            # --- PROCESS COLOR ---
            color_original = frame_to_bgr_image(color_frame)
            if color_original is None: continue

            # --- MANUAL ALIGNMENT (The Fix) ---
            # Crop center of color to match depth aspect ratio, then resize.
            color_aligned = manual_align_color_to_depth(color_original, width_d, height_d)

            # --- YOLO SEGMENTATION ---
            # Run on aligned image so mask fits depth perfectly
            results = model.predict(color_aligned, verbose=False, imgsz=640, conf=0.3)
            
            combined_mask = np.zeros((height_d, width_d), dtype=np.uint8)

            if results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                for mask in masks_data:
                    mask_resized = cv2.resize(mask, (width_d, height_d))
                    combined_mask = np.maximum(combined_mask, (mask_resized > 0.5).astype(np.uint8))
            elif results[0].boxes is not None and len(results[0].boxes) > 0:
                 for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(combined_mask, (x1, y1), (x2, y2), 1, -1)

            binary_mask = combined_mask.astype(np.uint8)
            background_mask = (binary_mask == 0)

            # --- APPLY MASKS ---
            masked_color = cv2.bitwise_and(color_aligned, color_aligned, mask=binary_mask)
            masked_raw_depth = cv2.bitwise_and(depth_raw_mm, depth_raw_mm, mask=binary_mask)

            # --- VISUALIZATION ---
            trtm_8bit, table_z = apply_trtm_shading(depth_raw_mm, background_mask)
            trtm_display_bgr = cv2.cvtColor(trtm_8bit, cv2.COLOR_GRAY2BGR)

            combined_display = np.hstack((color_aligned, trtm_display_bgr))

            # --- DRAW INFO ---
            if table_z > 0:
                cv2.putText(combined_display, f"Table Z: {table_z/10:.1f} cm", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- SNAPSHOT ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                os.makedirs(save_folder, exist_ok=True)

                final_color = center_crop_and_pad(masked_color, CLOTH_ROI_SIZE, TARGET_CANVAS_SIZE, bg_color=0)

                final_depth_raw = center_crop_and_pad(masked_raw_depth, CLOTH_ROI_SIZE, TARGET_CANVAS_SIZE, bg_color=0)
                
                final_depth_trtm = center_crop_and_pad(trtm_8bit, CLOTH_ROI_SIZE, TARGET_CANVAS_SIZE, bg_color=255)

                # Save Masked Raw Depth (Background=0) & Masked Aligned Color
                fname_color = os.path.join(save_folder, f"color_{frame_count:05d}.png")
                fname_raw = os.path.join(save_folder, f"depth_raw_{frame_count:05d}.png")
                fname_trtm = os.path.join(save_folder, f"depth_trtm_{frame_count:05d}.png")

                cv2.imwrite(fname_color, final_color)
                cv2.imwrite(fname_raw, final_depth_raw)
                cv2.imwrite(fname_trtm, final_depth_trtm)

                print(f"[{frame_count}] Saved")
                frame_count += 1
                show_flash = 5

            if show_flash > 0:
                overlay = combined_display.copy()
                cv2.rectangle(overlay, (0, 0), (combined_display.shape[1], combined_display.shape[0]), (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.5, combined_display, 0.5, 0, combined_display)
                show_flash -= 1

            cv2.imshow("Orbbec Femto Bolt TRTM (Manual Align)", combined_display)

            if key == ord('q') or key == ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()