import pandas as pd
import os
import shutil

frames_root = "frames"        
annotations_csv = "scand_manual_annotations.csv"
output_root = "labeled_frames"
os.makedirs(output_root, exist_ok=True)

# Read annotations
ann = pd.read_csv(annotations_csv)

for _, row in ann.iterrows():
    video = os.path.splitext(row["video_file"])[0]
    start_t = float(row["start_time_s"])
    end_t = float(row["end_time_s"])
    label = row["social_label"]

    frame_dir = os.path.join(frames_root, video)
    map_csv = os.path.join(frame_dir, "frame_index_map.csv")

    if not os.path.exists(map_csv):
        print(f"⚠ Missing mapping for {video}, skipping.")
        continue

    map_df = pd.read_csv(map_csv)

    selected = map_df[(map_df["time_sec"] >= start_t) & (map_df["time_sec"] <= end_t)]

    segment_name = f"{video}_{int(start_t)}-{int(end_t)}_{label.replace(' ', '_')}"
    segment_dir = os.path.join(output_root, segment_name)
    os.makedirs(segment_dir, exist_ok=True)

    for new_id in selected["new_frame_id"]:
        src = os.path.join(frame_dir, f"{int(new_id):06d}.jpg")
        dst = os.path.join(segment_dir, f"{int(new_id):06d}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"✅ {segment_name}: {len(selected)} frames labeled")

print("🎉 All segments processed! Check:", output_root)
