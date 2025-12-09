import cv2
import numpy as np
import os
import argparse
import glob

# --- CONFIGURATION MATCHING TRTM PAPER ---
TARGET_CANVAS_SIZE = 720   # Total image size
CLOTH_ROI_SIZE = 480       # Size of the cloth area
FLAT_PIXEL_VAL = 192       # Baseline value for flat cloth
PIXEL_PER_MM = 2           # 1cm (10mm) = 20 pixel val -> 2 pixel val per 1mm
BACKGROUND_COLOR = 255     # White background (Depth)
RGB_BACKGROUND_COLOR = 0   # Black background for RGB padding (standard for CV)

def process_depth_image(depth_raw, table_depth_mm):
    """
    Converts raw 16-bit depth (mm) to the 8-bit TRTM format.
    """
    # 1. Create the Height Map (Distance from Table)
    height_map_mm = table_depth_mm - depth_raw
    height_map_mm = np.where(height_map_mm < -10, 0, height_map_mm) 

    # 2. Convert Height (mm) to Pixel Intensity
    processed = FLAT_PIXEL_VAL + (height_map_mm * PIXEL_PER_MM)

    # 3. Handle Background (Threshold 5mm)
    mask_bg = height_map_mm < 5 
    processed[mask_bg] = BACKGROUND_COLOR

    # 4. Clip and Convert to 8-bit
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    return processed

def center_crop_and_pad(img, target_roi=480, target_canvas=720, is_rgb=False):
    """
    Crops the center 480x480 from input, then pads to 720x720.
    Handles both Grayscale (2D) and RGB (3D) arrays.
    """
    h, w = img.shape[:2]
    bg_val = RGB_BACKGROUND_COLOR if is_rgb else BACKGROUND_COLOR
    
    # 1. Handle inputs smaller than ROI (Pre-padding)
    if w < target_roi or h < target_roi:
        pad_w = max(0, target_roi - w)
        pad_h = max(0, target_roi - h)
        img = cv2.copyMakeBorder(
            img, 
            pad_h//2, pad_h-pad_h//2, 
            pad_w//2, pad_w-pad_w//2, 
            cv2.BORDER_CONSTANT, 
            value=(bg_val, bg_val, bg_val) if is_rgb else bg_val
        )
        h, w = img.shape[:2] # Update dimensions

    # 2. Center Crop (Extract ROI)
    start_x = max(0, (w - target_roi) // 2)
    start_y = max(0, (h - target_roi) // 2)
    end_x = start_x + target_roi
    end_y = start_y + target_roi
    
    cropped = img[start_y:end_y, start_x:end_x]

    # 3. Pad to Target Canvas
    if is_rgb:
        final_canvas = np.full((target_canvas, target_canvas, 3), bg_val, dtype=np.uint8)
    else:
        final_canvas = np.full((target_canvas, target_canvas), bg_val, dtype=np.uint8)
    
    # Calculate offset to center the ROI in the Canvas
    offset_x = (target_canvas - target_roi) // 2
    offset_y = (target_canvas - target_roi) // 2
    
    if is_rgb:
        final_canvas[offset_y:offset_y+target_roi, offset_x:offset_x+target_roi, :] = cropped
    else:
        final_canvas[offset_y:offset_y+target_roi, offset_x:offset_x+target_roi] = cropped
    
    return final_canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Folder containing raw images")
    parser.add_argument("--output", type=str, default="trtm_processed_data", help="Output folder")
    parser.add_argument("--manual_table_depth", type=int, default=None, help="Force a table depth in mm (e.g. 850)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input, "depth_*.png")))
    if not files:
        print("No depth images found! Make sure filenames start with 'depth_'")
        return

    print(f"Found {len(files)} image sets. Processing Depth AND Color...")

    for f in files:
        # --- PROCESS DEPTH ---
        depth_raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if depth_raw is None: continue

        # Auto-detect table depth if not provided
        if args.manual_table_depth:
            table_depth = args.manual_table_depth
        else:
            valid_pixels = depth_raw[depth_raw > 0]
            if len(valid_pixels) == 0: continue
            table_depth = np.percentile(valid_pixels, 98)

        # Process Depth (Normalize -> Crop -> Pad)
        trtm_depth = process_depth_image(depth_raw, table_depth)
        final_depth = center_crop_and_pad(trtm_depth, CLOTH_ROI_SIZE, TARGET_CANVAS_SIZE, is_rgb=False)

        # Save Depth
        filename_depth = os.path.basename(f)
        file_id = filename_depth.split(".")[0].split("_")[-1]
        cv2.imwrite(os.path.join(args.output, f'{file_id}.real_depth.png'), final_depth)

        # --- PROCESS COLOR ---
        # Infer color filename: depth_00001.png -> color_00001.png
        color_path = f.replace("depth_", "color_")
        
        if os.path.exists(color_path):
            color_raw = cv2.imread(color_path) # Reads as BGR by default (correct for OpenCV)
            
            # Process Color (Crop -> Pad ONLY) - No depth normalization needed
            final_color = center_crop_and_pad(color_raw, CLOTH_ROI_SIZE, TARGET_CANVAS_SIZE, is_rgb=True)
            
            filename_color = os.path.basename(color_path)
            cv2.imwrite(os.path.join(args.output, f'{file_id}.real_color.png'), final_color)
            print(f"Processed: {filename_depth} & {filename_color} | Table: {int(table_depth)}mm")
        else:
            print(f"Processed: {filename_depth} (No matching color file found)")

    print(f"\nDone! All processed images saved to: {args.output}")

if __name__ == "__main__":
    main()