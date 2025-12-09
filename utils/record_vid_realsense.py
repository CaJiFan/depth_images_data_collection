import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
WIDTH = 640
HEIGHT = 480
FPS = 30

def main():
    # 1. Setup Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Auto-resolve resolution (matches your hardware capability)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)

    print("Starting Camera...")
    try:
        pipeline.start(config)
    except:
        # Fallback for USB 2.0 dongles
        config.enable_all_streams()
        pipeline.start(config)

    # 2. Setup Output Folder
    # We create a new folder for every session to keep data clean
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_{timestamp}"
    
    is_recording = False
    frame_count = 0

    print(f"\nReady. Data will be saved to folder: {save_folder}")
    print("Controls:")
    print("  [S] - Start/Stop Saving Frames")
    print("  [Q] - Quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 3. Get Raw Data
            # This is the RAW 16-bit data (0-65535mm) TRTM needs
            depth_image_raw = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 4. Create Visualization (Grayscale)
            # We scale it down to 8-bit just for YOUR eyes (Display only)
            # alpha=0.03 scales the pixel values so objects 1-2m away are visible
            depth_display = cv2.convertScaleAbs(depth_image_raw, alpha=0.03)
            
            # Make it 3-channel grayscale so we can stack it with color
            depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

            # Resize color to match depth if necessary
            if depth_display.shape != color_image.shape:
                color_image = cv2.resize(color_image, (depth_display.shape[1], depth_display.shape[0]))

            # Stack images for display
            combined_display = np.hstack((color_image, depth_display))

            # 5. Saving Logic
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                is_recording = not is_recording
                if is_recording:
                    os.makedirs(save_folder, exist_ok=True)
                    print(f"[*] RECORDING STARTED... Saving to {save_folder}")
                else:
                    print(f"[*] RECORDING STOPPED. Saved {frame_count} frames.")

            if is_recording:
                # TRTM likely expects separate depth and color images
                # We save the RAW 16-bit depth (crucial for valid data)
                depth_filename = os.path.join(save_folder, f"depth_{frame_count:05d}.png")
                color_filename = os.path.join(save_folder, f"color_{frame_count:05d}.png")
                
                # Save Color
                cv2.imwrite(color_filename, color_image)
                
                # Save Raw Depth (Lossless PNG)
                # Note: This will look very dark/black in standard photo viewers 
                # because the values are small numbers (millimeters), but the DATA is correct.
                cv2.imwrite(depth_filename, depth_image_raw)
                
                frame_count += 1
                
                # Visual Indicator (Red Circle)
                cv2.circle(combined_display, (30, 30), 10, (0, 0, 255), -1)

            cv2.imshow('TRTM Data Collector (Gray = Depth)', combined_display)

            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()