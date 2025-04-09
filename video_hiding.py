import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim

def embed_video(cover_video, secret_video, stego_output, alpha=0.06):
    """Embed a faint secret video overlay onto the cover video with performance metrics and error analysis."""
    
    # Initialize performance metrics
    metrics = {
        'start_time': time.time(),
        'total_frames_processed': 0,
        'cover_frames_used': 0,
        'secret_frames_used': 0,
        'secret_frames_reused': 0,
        'resized_frames': 0,
        'frame_processing_times': [],
        'mse_errors': [],
        'psnr_values': [],
        'ssim_values': [],
        'frame_differences': []
    }

    # Check if files exist
    if not os.path.exists(cover_video):
        print(f" Error: Cover video '{cover_video}' not found!")
        return
    if not os.path.exists(secret_video):
        print(f" Error: Secret video '{secret_video}' not found!")
        return

    cap_cover = cv2.VideoCapture(cover_video)
    cap_secret = cv2.VideoCapture(secret_video)

    # Check if videos opened correctly
    if not cap_cover.isOpened():
        print(f" Error: Could not open cover video: {cover_video}")
        return
    if not cap_secret.isOpened():
        print(f" Error: Could not open secret video: {secret_video}")
        return

    # Get video properties
    frame_width = int(cap_cover.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_cover.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cover = cap_cover.get(cv2.CAP_PROP_FPS)
    fps_secret = cap_secret.get(cv2.CAP_PROP_FPS)
    cover_frame_count = int(cap_cover.get(cv2.CAP_PROP_FRAME_COUNT))
    secret_frame_count = int(cap_secret.get(cv2.CAP_PROP_FRAME_COUNT))
    cover_duration = cover_frame_count / fps_cover
    secret_duration = secret_frame_count / fps_secret

    print("\n=== Video Properties ===")
    print(f" Cover Video: {cover_video}")
    print(f"  - Resolution: {frame_width}x{frame_height}")
    print(f"  - Frames: {cover_frame_count}")
    print(f"  - FPS: {fps_cover:.2f}")
    print(f"  - Duration: {cover_duration:.2f} sec")
    print(f" Secret Video: {secret_video}")
    print(f"  - Frames: {secret_frame_count}")
    print(f"  - FPS: {fps_secret:.2f}")
    print(f"  - Duration: {secret_duration:.2f} sec")

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(stego_output, fourcc, fps_cover, (frame_width, frame_height))

    print("\n=== Processing Started ===")
    while cap_cover.isOpened():
        frame_start_time = time.time()
        
        ret_cover, frame_cover = cap_cover.read()
        if not ret_cover:
            break  # Stop if cover video ends
            
        metrics['cover_frames_used'] += 1
        
        ret_secret, frame_secret = cap_secret.read()
        if not ret_secret:
            # If secret video is shorter, rewind it
            cap_secret.set(cv2.CAP_PROP_POS_FRAMES, 0)
            metrics['secret_frames_reused'] += 1
            ret_secret, frame_secret = cap_secret.read()
            if not ret_secret:
                frame_secret = np.zeros_like(frame_cover)
        else:
            metrics['secret_frames_used'] += 1

        # Resize secret frame if needed
        if frame_secret.shape[0] != frame_height or frame_secret.shape[1] != frame_width:
            frame_secret = cv2.resize(frame_secret, (frame_width, frame_height))
            metrics['resized_frames'] += 1

        # Blend the frames
        stego_frame = cv2.addWeighted(frame_cover, 1.0, frame_secret, alpha, 0)

        # Calculate error metrics (every 10 frames for performance)
        if metrics['total_frames_processed'] % 10 == 0:
            # Convert to float32 for accurate calculations
            frame_cover_float = frame_cover.astype('float32')
            stego_frame_float = stego_frame.astype('float32')
            
            # Calculate MSE (Mean Squared Error)
            mse = np.mean((frame_cover_float - stego_frame_float) ** 2)
            metrics['mse_errors'].append(mse)
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            if mse == 0:
                psnr = 100  # Perfect reconstruction
            else:
                psnr = 10 * np.log10((255**2) / mse)
            metrics['psnr_values'].append(psnr)
            
            # Calculate SSIM (Structural Similarity) per channel and average
            ssim_values = []
            for channel in range(3):  # Process each color channel (BGR)
                ssim_val = ssim(
                    frame_cover_float[:,:,channel], 
                    stego_frame_float[:,:,channel],
                    data_range=stego_frame_float[:,:,channel].max()-stego_frame_float[:,:,channel].min()
                )
                ssim_values.append(ssim_val)
            metrics['ssim_values'].append(np.mean(ssim_values))
            
            # Calculate average pixel difference
            diff = cv2.absdiff(frame_cover, stego_frame)
            metrics['frame_differences'].append(np.mean(diff))

        # Write frame to output video
        out.write(stego_frame)

        metrics['total_frames_processed'] += 1
        metrics['frame_processing_times'].append(time.time() - frame_start_time)
        
        current_frame = int(cap_cover.get(cv2.CAP_PROP_POS_FRAMES))
        print(f" Processed frame {current_frame}/{cover_frame_count}", end='\r')

    # Calculate performance metrics
    total_time = time.time() - metrics['start_time']
    avg_frame_time = np.mean(metrics['frame_processing_times']) * 1000
    max_frame_time = np.max(metrics['frame_processing_times']) * 1000
    min_frame_time = np.min(metrics['frame_processing_times']) * 1000
    processing_rate = metrics['total_frames_processed'] / total_time
    
    # Calculate error metrics averages
    avg_mse = np.mean(metrics['mse_errors']) if metrics['mse_errors'] else 0
    avg_psnr = np.mean(metrics['psnr_values']) if metrics['psnr_values'] else 0
    avg_ssim = np.mean(metrics['ssim_values']) if metrics['ssim_values'] else 0
    avg_diff = np.mean(metrics['frame_differences']) if metrics['frame_differences'] else 0

    print("\n\n=== Processing Complete ===")
    print(f" Output file: {stego_output}")
    print(f" Total processing time: {total_time:.2f} seconds")
    print(f" Average frame processing time: {avg_frame_time:.2f} ms")
    print(f" Max frame time: {max_frame_time:.2f} ms | Min frame time: {min_frame_time:.2f} ms")
    print(f" Processing rate: {processing_rate:.2f} frames/sec")
    
    print("\n=== Frame Usage Statistics ===")
    print(f" Total frames processed: {metrics['total_frames_processed']}")
    print(f" Cover frames used: {metrics['cover_frames_used']}")
    print(f" Secret frames used: {metrics['secret_frames_used']}")
    print(f" Secret frames reused (looped): {metrics['secret_frames_reused']}")
    print(f" Frames resized: {metrics['resized_frames']}")
    
    print("\n=== Error Metrics ===")
    print(f" Average MSE (Mean Squared Error): {avg_mse:.2f}")
    print(f" Average PSNR (Peak Signal-to-Noise Ratio): {avg_psnr:.2f} dB")
    print(f" Average SSIM (Structural Similarity): {avg_ssim:.4f}")
    print(f" Average pixel difference: {avg_diff:.2f}")
    
    print("\n=== Error Interpretation ===")
    print(" MSE: Lower is better (0 = perfect)")
    print(" PSNR: Higher is better (>30 dB = good, >40 dB = excellent)")
    print(" SSIM: Closer to 1 is better (1 = perfect)")
    print(f" The current alpha ({alpha}) results in:")
    print(f" - {'Minimal' if avg_diff < 5 else 'Moderate' if avg_diff < 15 else 'Significant'} visual differences")
    print(f" - {'Excellent' if avg_psnr > 40 else 'Good' if avg_psnr > 30 else 'Fair'} quality preservation")

    cap_cover.release()
    cap_secret.release()
    out.release()

    print("\nStego video successfully created with comprehensive error analysis!")

# Run the embedding function
embed_video("cover.mp4", "secret.mp4", "stego.avi", alpha=0.06)