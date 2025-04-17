import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from skimage.metrics import structural_similarity as ssim
import time

def encrypt_file(input_file, output_file, password):
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32)
    cipher = AES.new(key, AES.MODE_GCM)
    with open(input_file, 'rb') as f:
        data = f.read()
    ciphertext, tag = cipher.encrypt_and_digest(data)
    with open(output_file, 'wb') as f:
        f.write(salt + cipher.nonce + tag + ciphertext)

def decrypt_file(encrypted_file, output_file, password):
    with open(encrypted_file, 'rb') as f:
        salt = f.read(16)
        nonce = f.read(16)
        tag = f.read(16)
        ciphertext = f.read()
    key = PBKDF2(password, salt, dkLen=32)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        data = cipher.decrypt_and_verify(ciphertext, tag)
        with open(output_file, 'wb') as f:
            f.write(data)
        return True
    except ValueError:
        return False

def embed_video(cover_video, secret_video, stego_output, alpha=0.04):
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

    cap_cover = cv2.VideoCapture(cover_video)
    cap_secret = cv2.VideoCapture(secret_video)

    if not cap_cover.isOpened() or not cap_secret.isOpened():
        return "Error: Couldn't open one or both videos.", metrics

    frame_width = int(cap_cover.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_cover.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cover = cap_cover.get(cv2.CAP_PROP_FPS)
    fps_secret = cap_secret.get(cv2.CAP_PROP_FPS)
    cover_frame_count = int(cap_cover.get(cv2.CAP_PROP_FRAME_COUNT))
    secret_frame_count = int(cap_secret.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(stego_output, cv2.VideoWriter_fourcc(*'XVID'), fps_cover, (frame_width, frame_height))

    while cap_cover.isOpened():
        frame_start = time.time()
        ret_cover, frame_cover = cap_cover.read()
        if not ret_cover:
            break
        metrics['cover_frames_used'] += 1
        ret_secret, frame_secret = cap_secret.read()
        if not ret_secret:
            cap_secret.set(cv2.CAP_PROP_POS_FRAMES, 0)
            metrics['secret_frames_reused'] += 1
            ret_secret, frame_secret = cap_secret.read()
            if not ret_secret:
                frame_secret = np.zeros_like(frame_cover)
        else:
            metrics['secret_frames_used'] += 1

        if frame_secret.shape[:2] != (frame_height, frame_width):
            frame_secret = cv2.resize(frame_secret, (frame_width, frame_height))
            metrics['resized_frames'] += 1

        stego_frame = cv2.addWeighted(frame_cover, 1.0, frame_secret, alpha, 0)
        out.write(stego_frame)

        if metrics['total_frames_processed'] % 10 == 0:
            fc = frame_cover.astype('float32')
            sf = stego_frame.astype('float32')
            mse = np.mean((fc - sf) ** 2)
            metrics['mse_errors'].append(mse)
            psnr = 100 if mse == 0 else 10 * np.log10((255**2) / mse)
            metrics['psnr_values'].append(psnr)
            ssim_avg = np.mean([
                ssim(fc[:, :, i], sf[:, :, i], data_range=np.ptp(sf[:, :, i]))
                for i in range(3)
            ])
            metrics['ssim_values'].append(ssim_avg)
            metrics['frame_differences'].append(np.mean(cv2.absdiff(frame_cover, stego_frame)))

        metrics['total_frames_processed'] += 1
        metrics['frame_processing_times'].append(time.time() - frame_start)

    cap_cover.release()
    cap_secret.release()
    out.release()

    total_time = time.time() - metrics['start_time']
    avg_time = np.mean(metrics['frame_processing_times']) * 1000
    report = f"""
=== Video Properties ===
 Cover Video: {os.path.basename(cover_video)}
  - Resolution: {frame_width}x{frame_height}
  - Frames: {cover_frame_count}
  - FPS: {fps_cover:.2f}
  - Duration: {cover_frame_count/fps_cover:.2f} sec
 Secret Video: {os.path.basename(secret_video)}
  - Frames: {secret_frame_count}
  - FPS: {fps_secret:.2f}
  - Duration: {secret_frame_count/fps_secret:.2f} sec

=== Processing Complete ===
 Output file: {stego_output}
 Total processing time: {total_time:.2f} seconds
 Average frame processing time: {avg_time:.2f} ms
 Max frame time: {max(metrics['frame_processing_times']) * 1000:.2f} ms | Min frame time: {min(metrics['frame_processing_times']) * 1000:.2f} ms
 Processing rate: {metrics['total_frames_processed'] / total_time:.2f} frames/sec

=== Frame Usage Statistics ===
 Total frames processed: {metrics['total_frames_processed']}
 Cover frames used: {metrics['cover_frames_used']}
 Secret frames used: {metrics['secret_frames_used']}
 Secret frames reused (looped): {metrics['secret_frames_reused']}
 Frames resized: {metrics['resized_frames']}

=== Error Metrics ===
 Average MSE (Mean Squared Error): {np.mean(metrics['mse_errors']):.2f}
 Average PSNR (Peak Signal-to-Noise Ratio): {np.mean(metrics['psnr_values']):.2f} dB
 Average SSIM (Structural Similarity): {np.mean(metrics['ssim_values']):.4f}
 Average pixel difference: {np.mean(metrics['frame_differences']):.2f}

=== Error Interpretation ===
 MSE: Lower is better (0 = perfect)
 PSNR: Higher is better (>30 dB = good, >40 dB = excellent)
 SSIM: Closer to 1 is better (1 = perfect)
 The current alpha ({alpha}) results in:
 - {"Minimal" if np.mean(metrics['frame_differences']) < 5 else "Moderate" if np.mean(metrics['frame_differences']) < 15 else "Significant"} visual differences
 - {"Excellent" if np.mean(metrics['psnr_values']) > 40 else "Good" if np.mean(metrics['psnr_values']) > 30 else "Fair"} quality preservation

Stego video successfully created with comprehensive error analysis!
"""
    return report, metrics

# GUI
class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video In Video Hiding Using Transform Domain By Group 14")
        self.root.geometry("700x600")

        self.cover_path = ""
        self.secret_path = ""
        self.output_path = "stego.avi"
        self.encrypted_secret_path = "stego_encrypted.bin"
        self.temp_decrypted_path = "secret..mp4"

        self.build_widgets()

    def build_widgets(self):
        tk.Button(self.root, text="Select Cover Video", command=self.select_cover).pack(pady=5)
        tk.Button(self.root, text="Select Secret Video", command=self.select_secret).pack(pady=5)

        tk.Label(self.root, text="Set Encryption Key:").pack()
        self.entry_encrypt = tk.Entry(self.root, show="*")
        self.entry_encrypt.pack()

        tk.Button(self.root, text="Process & Encrypt Stego Video", command=self.encrypt_secret).pack(pady=5)

        tk.Label(self.root, text="Enter Key to Decrypt Stego:").pack()
        self.entry_decrypt = tk.Entry(self.root, show="*")
        self.entry_decrypt.pack()

        tk.Button(self.root, text="Decrypt Stego", command=self.decrypt_and_embed).pack(pady=5)

        self.result_text = scrolledtext.ScrolledText(self.root, width=85, height=20)
        self.result_text.pack(pady=10)

    def select_cover(self):
        self.cover_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.cover_path:
            messagebox.showinfo("Selected", f"Cover: {self.cover_path}")

    def select_secret(self):
        self.secret_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.secret_path:
            messagebox.showinfo("Selected", f"Secret: {self.secret_path}")

    def encrypt_secret(self):
        if not self.secret_path or not self.entry_encrypt.get():
            messagebox.showerror("Missing Info", "Select a secret video and set a password.")
            return
        encrypt_file(self.secret_path, self.encrypted_secret_path, self.entry_encrypt.get())
        messagebox.showinfo("Encrypted", "Stego video encrypted.")

    def decrypt_and_embed(self):
        if not self.cover_path or not self.entry_decrypt.get():
            messagebox.showerror("Missing Info", "Select a cover video and enter decryption key.")
            return

        success = decrypt_file(self.encrypted_secret_path, self.temp_decrypted_path, self.entry_decrypt.get())
        if not success:
            messagebox.showerror("Failed", "Wrong password or corrupted file.")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "\n=== Processing Started ===\n")
        self.root.update()

        report, metrics = embed_video(self.cover_path, self.temp_decrypted_path, self.output_path)
        os.remove(self.temp_decrypted_path)

        self.result_text.insert(tk.END, f"\nProcessed frame {metrics['total_frames_processed']}/{metrics['cover_frames_used']}\n")
        self.result_text.insert(tk.END, report)

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()
