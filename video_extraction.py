import cv2
import numpy as np
import os

def extract_video_perfect(stego_video, cover_video, extracted_output, alpha=0.06):
    """Extract the secret video perfectly by subtracting the cover video."""
    
    # Check if files exist
    if not os.path.exists(stego_video):
        print(f" Error: Stego video '{stego_video}' not found!")
        return
    if not os.path.exists(cover_video):
        print(f" Error: Cover video '{cover_video}' not found!")
        return

    cap_stego = cv2.VideoCapture(stego_video)
    cap_cover = cv2.VideoCapture(cover_video)

    # Check if videos opened correctly
    if not cap_stego.isOpened():
        print(f" Error: Could not open stego video: {stego_video}")
        return
    if not cap_cover.isOpened():
        print(f" Error: Could not open cover video: {cover_video}")
        return

    # Get video properties
    frame_width = int(cap_stego.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_stego.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_stego.get(cv2.CAP_PROP_FPS)
    stego_frame_count = int(cap_stego.get(cv2.CAP_PROP_FRAME_COUNT))
    cover_frame_count = int(cap_cover.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f" Stego Video: {stego_video} | Frames: {stego_frame_count} | FPS: {fps}")
    print(f" Cover Video: {cover_video} | Frames: {cover_frame_count}")

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(extracted_output, fourcc, fps, (frame_width, frame_height))

    while cap_stego.isOpened() and cap_cover.isOpened():
        ret_stego, frame_stego = cap_stego.read()
        ret_cover, frame_cover = cap_cover.read()
        
        if not ret_stego or not ret_cover:
            break  # Stop if either video ends

        # Resize cover frame if needed (must match stego dimensions)
        if frame_cover.shape[0] != frame_height or frame_cover.shape[1] != frame_width:
            frame_cover = cv2.resize(frame_cover, (frame_width, frame_height))

        # Convert to float32 for accurate subtraction
        frame_stego_float = frame_stego.astype(np.float32)
        frame_cover_float = frame_cover.astype(np.float32)

        # Extract secret: (stego - cover) / alpha
        frame_secret = (frame_stego_float - frame_cover_float) / alpha

        # Clip to valid range [0, 255] and convert back to uint8
        frame_secret = np.clip(frame_secret, 0, 255).astype(np.uint8)

        # Write frame to output video
        out.write(frame_secret)

        current_frame = int(cap_stego.get(cv2.CAP_PROP_POS_FRAMES))
        print(f" Processed frame {current_frame}/{stego_frame_count}", end='\r')

    cap_stego.release()
    cap_cover.release()
    out.release()

    print(f"\nSecret video perfectly extracted: {extracted_output}")

# Run the extraction function
extract_video_perfect("stego.avi", "cover.mp4", "extracted_secret_perfect.avi", alpha=0.06)