"""
Test script for LLaVA processing at 1Hz on video input.
This simulates real-time social navigation decision making by processing 
video frames at 1 frame per second and measuring LLaVA's response time.
"""

import cv2
import ollama
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime


def process_frame_with_llava(frame, prompt, model='llava'):
    """
    Process a single frame with LLaVA and return the response and processing time.
    
    Args:
        frame: OpenCV image frame
        prompt: Text prompt to send with the image
        model: Ollama model name (default: 'llava')
    
    Returns:
        tuple: (response_text, processing_time_seconds)
    """
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Measure processing time
    start_time = time.time()
    
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }]
        )
        processing_time = time.time() - start_time
        return response['message']['content'], processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        return f"Error: {str(e)}", processing_time


def test_video_1hz(video_path, output_dir=None, max_frames=None, social_nav_prompt=False):
    """
    Process video at 1Hz rate and measure LLaVA's performance.
    
    Args:
        video_path: Path to input video file
        output_dir: Optional directory to save annotated frames
        max_frames: Maximum number of frames to process (None for all)
        social_nav_prompt: Use social navigation specific prompt
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")
    print(f"Processing at 1Hz (every {int(fps)} frames)")
    print(f"{'='*80}\n")
    
    # Choose prompt based on mode
    if social_nav_prompt:
        prompt = """You are a social navigation assistant for a mobile robot. Analyze this scene and provide:
1. People present and their activities
2. Potential obstacles or hazards
3. Suggested navigation action (move forward, slow down, stop, turn left/right, wait)
Keep response concise (max 5 sentences)."""
    else:
        prompt = "Describe what you see in this image, focusing on people, obstacles, and the environment. Keep it concise (max 4 sentences)."
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Processing stats
    processing_times = []
    frame_count = 0
    processed_count = 0
    frame_interval = int(fps) if fps > 0 else 1
    
    # Start processing
    start_time = datetime.now()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame (where N = fps for 1Hz)
            if frame_count % frame_interval == 0:
                print(f"\n--- Frame {frame_count} (t={frame_count/fps:.2f}s) ---")
                
                # Process with LLaVA
                response, proc_time = process_frame_with_llava(frame, prompt)
                processing_times.append(proc_time)
                processed_count += 1
                
                print(f"Processing time: {proc_time:.3f}s")
                print(f"Response: {response}")
                
                # Save annotated frame if output directory specified
                if output_dir:
                    # Add text overlay with timestamp and processing time
                    annotated_frame = frame.copy()
                    text = f"Frame: {frame_count} | Time: {proc_time:.2f}s"
                    cv2.putText(annotated_frame, text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    output_path = output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(output_path), annotated_frame)
                
                # Check if we've reached max frames
                if max_frames and processed_count >= max_frames:
                    print(f"\nReached maximum of {max_frames} frames to process.")
                    break
                
                # Wait to maintain 1Hz rate (if processing was faster than 1s)
                if proc_time < 1.0:
                    time.sleep(1.0 - proc_time)
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    
    finally:
        cap.release()
        
        # Print statistics
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("PROCESSING STATISTICS")
        print(f"{'='*80}")
        print(f"Frames processed: {processed_count}")
        print(f"Total elapsed time: {elapsed:.2f}s")
        print(f"Average processing time: {sum(processing_times)/len(processing_times):.3f}s")
        print(f"Min processing time: {min(processing_times):.3f}s")
        print(f"Max processing time: {max(processing_times):.3f}s")
        print(f"Actual processing rate: {processed_count/elapsed:.2f} Hz")
        
        # Check if 1Hz is achievable
        avg_time = sum(processing_times)/len(processing_times)
        if avg_time > 1.0:
            print(f"\n⚠️  WARNING: Average processing time ({avg_time:.3f}s) exceeds 1 second!")
            print(f"   Maximum achievable rate: ~{1/avg_time:.2f} Hz")
        else:
            print(f"\n✓ 1Hz processing is achievable (avg: {avg_time:.3f}s < 1.0s)")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test LLaVA video processing at 1Hz for social navigation'
    )
    parser.add_argument(
        'video_path',
        type=str,
        nargs='?',
        help='Path to input video file (or leave empty to use default)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save annotated frames'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum number of frames to process (default: all)'
    )
    parser.add_argument(
        '--social-nav',
        action='store_true',
        help='Use social navigation specific prompt'
    )
    parser.add_argument(
        '--list-videos',
        action='store_true',
        help='List available videos in datasets/videos directory'
    )
    
    args = parser.parse_args()
    
    # List videos if requested
    if args.list_videos:
        videos_dir = Path(__file__).parent.parent / 'datasets' / 'videos'
        if videos_dir.exists():
            print("\nAvailable videos:")
            for video in sorted(videos_dir.rglob('*.mp4')):
                print(f"  {video.relative_to(videos_dir.parent)}")
        else:
            print("No videos directory found")
        return
    
    # Determine video path
    if args.video_path:
        video_path = Path(args.video_path)
    else:
        # Use default video
        default_video = Path(__file__).parent.parent / 'datasets' / 'videos' / 'spot' / 'A_Spot_Library_Fountain_Tue_Nov_16_119.mp4'
        if not default_video.exists():
            print("Error: No video path provided and default video not found.")
            print("Use --list-videos to see available videos.")
            return
        video_path = default_video
        print(f"Using default video: {video_path.name}")
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Run test
    test_video_1hz(
        video_path,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        social_nav_prompt=args.social_nav
    )


if __name__ == '__main__':
    main()
