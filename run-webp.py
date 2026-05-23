import os
import glob
import cv2
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
# Target dimensions (mimicking the Jetson camera output)
TARGET_W = 512
TARGET_H = 288

# The exact compression parameters from your bandwidth test
WEBP_PARAMS = [int(cv2.IMWRITE_WEBP_QUALITY), 95]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize images and compress them to WebP."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to folder containing source images"
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to folder where processed images will be saved"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    SOURCE_FOLDER = args.input
    OUTPUT_FOLDER = args.output

    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. Gather all images from the source folder
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))
    
    if not image_paths:
        print(f"Error: No images found in '{SOURCE_FOLDER}'. Check your path.")
        return
        
    print(f"Found {len(image_paths)} images.")
    print(f"Simulating drone pipeline: Resizing to {TARGET_W}x{TARGET_H} and compressing to WebP-95...")
    print("-" * 50)

    success_count = 0
    total_original_kb = 0.0
    total_compressed_kb = 0.0

    for idx, path in enumerate(image_paths):

        orig_size_kb = os.path.getsize(path) / 1024.0

        # Read original image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}")
            continue
            
        # STEP 1: Hardware Resize Simulation
        # Using INTER_AREA as it is the mathematically superior method for downsampling
        scaled_frame = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        # STEP 2: Network Compression Simulation
        # We save it as a .webp file to introduce the exact same (minimal) compression artifacts
        # that the radio link will produce.
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_filename = f"{base_name}_simulated.webp"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save the image
        success = cv2.imwrite(output_path, scaled_frame, WEBP_PARAMS)
        if success:
            success_count += 1

            comp_size_kb = os.path.getsize(output_path) / 1024.0

            total_original_kb += orig_size_kb
            total_compressed_kb += comp_size_kb
            
        # Print progress every 50 images
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images...")

    # ==========================================
    # PRINT SUMMARY
    # ==========================================
    print("-" * 50)
    print(f"✅ Done! Successfully simulated {success_count} frames.")

    if success_count > 0:
        avg_orig = total_original_kb / success_count
        avg_comp = total_compressed_kb / success_count
        compression_ratio = avg_orig / avg_comp
        
        print("\n📊 BANDWIDTH METRICS:")
        print(f"Average Original Size   : {avg_orig:.2f} KB")
        print(f"Average Compressed Size : {avg_comp:.2f} KB")
        print(f"Average Bandwidth Saved : {(1 - (avg_comp/avg_orig)) * 100:.1f}%")
        print(f"Average Ratio           : {compression_ratio:.1f}x")

    print(f"You can now point your MiDaS inference script to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()