import pyrealsense2 as rs

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("No device connected")
        return

    for dev in devices:
        print(f"\nDevice: {dev.get_info(rs.camera_info.name)}")
        print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"USB Type: {dev.get_info(rs.camera_info.usb_type_descriptor)}") 
        # ^ This line is CRITICAL. If it says "2.1", you cannot do high-res.

        print("="*40)
        
        for sensor in dev.query_sensors():
            print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
            print("-" * 20)
            
            # Get all valid stream profiles for this sensor
            profiles = sensor.get_stream_profiles()
            
            # Filter and sort so we don't get 500 lines of text
            unique_profiles = set()
            for p in profiles:
                # We only care about Video streams (width/height/fps)
                if p.is_video_stream_profile():
                    v_profile = p.as_video_stream_profile()
                    name = p.stream_name()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()
                    fmt = p.format()
                    
                    # Store as a readable string
                    unique_profiles.add(f"{name}: {w}x{h} @ {fps}fps ({fmt})")

            # Print sorted list
            for entry in sorted(list(unique_profiles), reverse=True):
                print(entry)

if __name__ == "__main__":
    get_profiles()