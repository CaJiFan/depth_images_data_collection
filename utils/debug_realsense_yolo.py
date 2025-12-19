import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'yoloe-11l-seg.pt'

def main():
    print("1. Loading YOLO...")
    model = YOLO(MODEL_PATH)
    model.set_classes(["cloth", "towel"])
    print("2. Starting RealSense Pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Use generic config to ensure stream starts
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting pipeline: {e}")
        return

    print("3. Stream Started. Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to Numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --- DEBUG 1: Check Max Depth ---
            # If this is 0, your depth sensor isn't seeing anything
            max_depth = np.max(depth_image)
            
            # --- DEBUG 2: YOLO Inference ---
            # Run YOLO but DO NOT mask yet. Just draw boxes.
            results = model.predict(color_image, verbose=False, conf=0.25)
            
            # Plot the results on the image (Draws boxes + labels)
            # This returns a BGR numpy array with drawings
            annotated_frame = results[0].plot()

            # --- DEBUG 3: Depth Visualization ---
            # Use colormap so we can see ANY valid depth
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack images
            combined = np.hstack((annotated_frame, depth_colormap))

            cv2.imshow('Debug View (Left: YOLO, Right: Depth)', combined)
            
            # Print status every 30 frames (approx 1 sec)
            # This helps debug "Frozen" streams vs "Black" streams
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()