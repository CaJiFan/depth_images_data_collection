import pyrealsense2 as rs
import numpy as np
import cv2

# --- CONFIGURATION ---
WIDTH = 424
HEIGHT = 240
FPS = 30

def main():
    # 1. Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable Depth and Color streams
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # 2. Start the pipeline
    print("Starting camera... (Using sudo if this fails!)")
    profile = pipeline.start(config)

    # Helper to calculate scale (optional, but good for depth accuracy)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")

    try:
        print("Stream started. Press 'q' or 'ESC' to exit.")
        while True:
            # 3. Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # print(color_frame)
            if not depth_frame or not color_frame:
                continue

            # 4. Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 5. Visualise Depth: Apply ColorMap
            # The depth image is 16-bit (0-65535). We scale it down to 8-bit (0-255) for display.
            # alpha=0.03 is a scaling factor (adjust if image is too dark/bright)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 6. Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # 7. Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # 8. Exit condition
            key = cv2.waitKey(1)
            # Press 'q' or 'ESC' to close
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        # 9. Stop streaming nicely
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()