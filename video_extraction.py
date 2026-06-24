import cv2
import numpy as np
import os

def extract_video_perfect(stego_video, cover_video, extracted_output, alpha=0.04):
    """Extract the secret video perfectly by subtracting the cover video."""
    
    if not os.path.exists(stego_video):
        print(f"Error: Stego video '{stego_video}' not found!")
        return
    if not os.path.exists(cover_video):
        print(f"Error: Cover video '{cover_video}' not found!")
        return

    cap_stego = cv2.VideoCapture(stego_video)
    cap_cover = cv2.VideoCapture(cover_video)

    if not cap_stego.isOpened():
        print(f"Error: Could not open stego video: {stego_video}")
        return
    if not cap_cover.isOpened():
        print(f"Error: Could not open cover video: {cover_video}")
        return

    frame_width = int(cap_stego.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_stego.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_stego.get(cv2.CAP_PROP_FPS)
    stego_frame_count = int(cap_stego.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Stego Video: {stego_video} | Frames: {stego_frame_count} | FPS: {fps}")
    print(f"Cover Video: {cover_video}")
    print(f"Starting extraction... (Alpha: {alpha})")

    # Output extracted video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(extracted_output, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap_stego.isOpened() and cap_cover.isOpened():
        ret_stego, frame_stego = cap_stego.read()
        ret_cover, frame_cover = cap_cover.read()
        
        if not ret_stego or not ret_cover:
            break 

        if frame_cover.shape[0] != frame_height or frame_cover.shape[1] != frame_width:
            frame_cover = cv2.resize(frame_cover, (frame_width, frame_height))

        # Convert to float32 for accurate math
        frame_stego_float = frame_stego.astype(np.float32)
        frame_cover_float = frame_cover.astype(np.float32)

        # Extract secret: (stego - cover) / alpha
        frame_secret = (frame_stego_float - frame_cover_float) / alpha

        # Clip invalid pixels and convert back
        frame_secret = np.clip(frame_secret, 0, 255).astype(np.uint8)

        out.write(frame_secret)
        frame_count += 1
        print(f"Processed frame {frame_count}/{stego_frame_count}", end='\r')

    cap_stego.release()
    cap_cover.release()
    out.release()

    print(f"\nExtraction complete! Secret video saved to: {extracted_output}")

if __name__ == "__main__":
    # Ensure these filenames match the output from the hiding script exactly
    stego_file = "stego_output.mkv"
    cover_file = "cover.mp4" # Replace with your original cover video's exact name
    output_file = "extracted_secret.avi"
    
    extract_video_perfect(stego_file, cover_file, output_file, alpha=0.04)
