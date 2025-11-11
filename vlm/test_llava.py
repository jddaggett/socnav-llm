# Test script for ollama's LLaVA model with webcam input
# LLaVA is a novel end-to-end trained large multimodal model that combines a vision encoder 
# and Vicuna for general-purpose visual and language understanding.

import cv2
import ollama
import base64

# Set up a video capture object
cap = cv2.VideoCapture(0)  # 0 means use default camera

print("Capturing single frame...")
ret, frame = cap.read()

if ret:
    # Convert frame to base64 for Ollama
    # Adjust brightness and contrast before encoding
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 75    # Brightness control (0-100)
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Encode the adjusted frame to base64
    _, buffer = cv2.imencode('.jpg', adjusted_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Use Ollama's LLaVA model to describe the frame
    try:
        print("Analyzing image with LLaVA...")
        response = ollama.chat(
            model='llava',
            messages=[{
                'role': 'user',
                'content': 'Describe everything you see in this image. Do not exceed 4 sentences.',
                'images': [image_base64]
            }]
        )
        description = response['message']['content']
        print(f"LLaVA says: {description}")
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)

    # Show the adjusted frame (same as what's sent to LLaVA)
    cv2.imshow('Captured Frame (Brightness Enhanced)', adjusted_frame)
    print("Press any key in the image window to close...")
    cv2.waitKey(0)  # Wait for any key press
else:
    print("Failed to capture frame from camera")

# Release the video capture object and close any windows
cap.release()
cv2.destroyAllWindows()