import cv2
import os
import argparse
import csv
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 10)  # convert to ~10 fps extraction

    frame_idx = 0
    save_id = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # prepare mapping file
    map_csv_path = os.path.join(output_dir, "frame_index_map.csv")
    csv_file = open(map_csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["new_frame_id", "original_frame_id", "time_sec"])

    for _ in tqdm(range(total), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / fps
            filename = os.path.join(output_dir, f"{save_id:06d}.jpg")
            cv2.imwrite(filename, frame)
            writer.writerow([save_id, frame_idx, round(time_sec, 3)])
            save_id += 1

        frame_idx += 1

    cap.release()
    csv_file.close()
    print(f"✅ Done: {save_id} frames saved to {output_dir}")
    print(f"✅ Mapping saved to {map_csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="videos", help="Folder of .mp4 files")
    parser.add_argument("--output_root", default="frames", help="Where to store output frames")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    for filename in os.listdir(args.video_dir):
        if filename.lower().endswith(".mp4"):
            video_path = os.path.join(args.video_dir, filename)
            video_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(args.output_root, video_name)
            extract_frames(video_path, output_dir)

if __name__ == "__main__":
    main()
