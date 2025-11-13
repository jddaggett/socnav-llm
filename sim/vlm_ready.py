# vlm_ready.py

import json
import base64
import os
from tqdm import tqdm

def process_labeled_frames(input_json, output_dir="frames_base64"):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json) as f:
        data = json.load(f)

    for segment in tqdm(data, desc="Processing segments"):
        segment_name = segment["segment_name"]
        segment_dir = os.path.join(output_dir, segment_name)
        os.makedirs(segment_dir, exist_ok=True)

        for i, frame_path in enumerate(segment["frames"]):
            if not os.path.exists(frame_path):
                print(f"⚠ Frame missing: {frame_path}")
                continue
            with open(frame_path, "rb") as img_f:
                encoded = base64.b64encode(img_f.read()).decode("utf-8")
            out_file = os.path.join(segment_dir, f"{i:06d}.txt")
            with open(out_file, "w") as f_out:
                f_out.write(encoded)

    print("All frames processed.")

if __name__ == "__main__":
    input_json = "labeled_frames.json"  
    if not os.path.exists(input_json):
        print(f"JSON file not found: {input_json}")
    else:
        print(f"Found JSON file: {input_json}")
        process_labeled_frames(input_json)
        print(" Script finished.")