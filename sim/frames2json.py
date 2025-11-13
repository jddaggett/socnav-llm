import os
import json
from PIL import Image

frames_root = "labeled_frames"  
output_file = "labeled_frames.json"
frame_sample_interval = 5  

dataset = []

for segment_name in os.listdir(frames_root):
    segment_dir = os.path.join(frames_root, segment_name)
    if not os.path.isdir(segment_dir):
        continue
    
    frames = sorted(os.listdir(segment_dir))
    
    # 取最后一个下划线后面的作为 label
    video_part, label = segment_name.rsplit('_', 1)
    label = label.replace('_', ' ')  # 可选

    sampled_frames = frames[::frame_sample_interval]
    frame_paths = [os.path.join(segment_dir, f) for f in sampled_frames]
    
    dataset.append({
        "segment_name": segment_name,
        "label": label,
        "frames": frame_paths
    })

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ VLM dataset prepared: {len(dataset)} segments")
