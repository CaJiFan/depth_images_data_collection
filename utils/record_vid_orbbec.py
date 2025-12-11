import cv2
import numpy as np
import os
import time
from datetime import datetime
from pyorbbecsdk import *

# --- CONFIGURATION ---
ESC_KEY = 27
MIN_DEPTH = 20      # 20mm
MAX_DEPTH = 10000   # 10m

def main():
    config = Config()
    pipeline = Pipeline()

    # 1. Setup Depth Stream
    try:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profiles is None:
            print("No depth sensor found!")
            return
        # Get default profile (usually 640x576 or similar for Femto)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        print(f"Depth Profile: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")
    except Exception as e:
        print(f"Error enabling depth: {e}")
        return

    # 2. Setup Color Stream
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profiles is None:
            print("No color sensor found!")
            return
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        print(f"Color Profile: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps")
    except Exception as e:
        print(f"Error enabling color: {e}")
        return

    # 3. Start Pipeline
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        return

    # Setup Recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_orbbec_{timestamp}"
    is_recording = False
    frame_count = 0

    print(f"\nReady. Data will be saved to: {save_folder}")
    print("Controls:")
    print("  [S] - Start/Stop Saving Frames")
    print("  [Q] - Quit")

    try:
        while True:
            # Wait for frames (100ms timeout)
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            # --- GET DEPTH ---
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            # Convert to Numpy (Raw 16-bit)
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            
            # Apply scale if necessary (Femto Bolt usually outputs 1mm units directly, but good to be safe)
            # We keep it as uint16 for saving (mm)
            depth_raw_mm = depth_data # This is the data TRTM needs

            # --- GET COLOR ---
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            
            # Orbbec color handling (MJPG or RGB)
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            
            if color_frame.get_format() == OBFormat.MJPG:
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            elif color_frame.get_format() == OBFormat.RGB:
                color_image = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            elif color_frame.get_format() == OBFormat.BGR:
                color_image = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
            else:
                # Fallback or unknown format
                color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)

            if color_image is None:
                continue

            # --- VISUALIZATION ---
            # Create a grayscale depth view (similar to TRTM paper look)
            # Alpha=0.03 scales 0-8000mm to 0-255 roughly
            depth_display = cv2.convertScaleAbs(depth_raw_mm, alpha=0.03)
            depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

            # Resize Color to match Depth height for side-by-side display
            if color_image.shape[0] != depth_display.shape[0]:
                aspect_ratio = color_image.shape[1] / color_image.shape[0]
                new_w = int(depth_display.shape[0] * aspect_ratio)
                color_display = cv2.resize(color_image, (new_w, depth_display.shape[0]))
            else:
                color_display = color_image

            combined_display = np.hstack((color_display, depth_display))

            # --- RECORDING LOGIC ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                is_recording = not is_recording
                if is_recording:
                    os.makedirs(save_folder, exist_ok=True)
                    print(f"[*] RECORDING STARTED... Saving to {save_folder}")
                else:
                    print(f"[*] RECORDING STOPPED. Saved {frame_count} frames.")

            if is_recording:
                depth_filename = os.path.join(save_folder, f"depth_{frame_count:05d}.png")
                color_filename = os.path.join(save_folder, f"color_{frame_count:05d}.png")

                # Save Raw 16-bit Depth (Essential for TRTM)
                cv2.imwrite(depth_filename, depth_raw_mm)
                
                # Save Color
                cv2.imwrite(color_filename, color_image)

                frame_count += 1
                
                # Draw Red Dot on Display
                cv2.circle(combined_display, (30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("Orbbec TRTM Collector", combined_display)

            if key == ord('q') or key == ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()