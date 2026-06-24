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
        return "Error: Couldn't open one or both videos. Check file formats.", metrics

    frame_width = int(cap_cover.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_cover.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cover = cap_cover.get(cv2.CAP_PROP_FPS)
    fps_secret = cap_secret.get(cv2.CAP_PROP_FPS)
    cover_frame_count = int(cap_cover.get(cv2.CAP_PROP_FRAME_COUNT))
    secret_frame_count = int(cap_secret.get(cv2.CAP_PROP_FRAME_COUNT))

    # Using FFV1 lossless codec and .mkv to ensure extraction works perfectly
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(stego_output, fourcc, fps_cover, (frame_width, frame_height))

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

        # Alpha Blending
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
                ssim(fc[:, :, i], sf[:, :, i], data_range=np.ptp(sf[:, :, i]) or 1.0)
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
    avg_time = np.mean(metrics['frame_processing_times']) * 1000 if metrics['frame_processing_times'] else 0
    max_time = max(metrics['frame_processing_times']) * 1000 if metrics['frame_processing_times'] else 0
    min_time = min(metrics['frame_processing_times']) * 1000 if metrics['frame_processing_times'] else 0

    report = f"""
=== Video Properties ===
 Cover Video: {os.path.basename(cover_video)}
  - Resolution: {frame_width}x{frame_height}
  - Frames: {cover_frame_count}
  - FPS: {fps_cover:.2f}
 Secret Video: {os.path.basename(secret_video)}
  - Frames: {secret_frame_count}
  - FPS: {fps_secret:.2f}

=== Processing Complete ===
 Output file: {stego_output}
 Total processing time: {total_time:.2f} sec
 Avg frame processing time: {avg_time:.2f} ms
 Max frame time: {max_time:.2f} ms | Min frame time: {min_time:.2f} ms

=== Frame Usage Statistics ===
 Total frames processed: {metrics['total_frames_processed']}
 Frames resized: {metrics['resized_frames']}

=== Error Metrics ===
 Avg MSE: {np.mean(metrics['mse_errors']):.2f}
 Avg PSNR: {np.mean(metrics['psnr_values']):.2f} dB
 Avg SSIM: {np.mean(metrics['ssim_values']):.4f}

Stego video successfully created!
"""
    return report, metrics


class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video In Video Hiding")
        self.root.geometry("700x600")

        self.cover_path = ""
        self.secret_path = ""
        
        # Changed stego output to .mkv to support the lossless FFV1 codec
        self.output_path = "stego_output.mkv"
        self.encrypted_secret_path = "stego_encrypted.bin"

        self.build_widgets()

    def build_widgets(self):
        tk.Button(self.root, text="1. Select Cover Video", command=self.select_cover).pack(pady=5)
        tk.Button(self.root, text="2. Select Secret Video", command=self.select_secret).pack(pady=5)

        tk.Label(self.root, text="Set Encryption Key:").pack()
        self.entry_encrypt = tk.Entry(self.root, show="*")
        self.entry_encrypt.pack()

        tk.Button(self.root, text="3. Process & Encrypt Secret Video", command=self.encrypt_secret).pack(pady=5)

        tk.Label(self.root, text="Enter Key to Decrypt & Embed:").pack()
        self.entry_decrypt = tk.Entry(self.root, show="*")
        self.entry_decrypt.pack()

        tk.Button(self.root, text="4. Embed Stego Video", command=self.decrypt_and_embed).pack(pady=10)

        self.result_text = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.result_text.pack(pady=10)

    def select_cover(self):
        self.cover_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.cover_path:
            messagebox.showinfo("Selected", f"Cover: {os.path.basename(self.cover_path)}")

    def select_secret(self):
        self.secret_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if self.secret_path:
            messagebox.showinfo("Selected", f"Secret: {os.path.basename(self.secret_path)}")

    def encrypt_secret(self):
        if not self.secret_path or not self.entry_encrypt.get():
            messagebox.showerror("Error", "Select a secret video and enter a password first.")
            return
        
        self.result_text.insert(tk.END, "Encrypting secret video...\n")
        self.root.update()
        
        encrypt_file(self.secret_path, self.encrypted_secret_path, self.entry_encrypt.get())
        messagebox.showinfo("Success", "Secret video encrypted successfully.")
        self.result_text.insert(tk.END, "Encryption Complete.\n")

    def decrypt_and_embed(self):
        if not self.cover_path or not os.path.exists(self.encrypted_secret_path):
            messagebox.showerror("Error", "Missing cover video or encrypted file. Complete previous steps.")
            return
        if not self.entry_decrypt.get():
            messagebox.showerror("Error", "Enter the decryption key.")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Decrypting file...\n")
        self.root.update()

        # Dynamically preserve the original extension of the secret video
        _, file_extension = os.path.splitext(self.secret_path)
        temp_decrypted_path = f"temp_secret_video{file_extension}"

        success = decrypt_file(self.encrypted_secret_path, temp_decrypted_path, self.entry_decrypt.get())
        if not success:
            messagebox.showerror("Failed", "Wrong password or corrupted encrypted file.")
            if os.path.exists(temp_decrypted_path):
                os.remove(temp_decrypted_path)
            return

        self.result_text.insert(tk.END, "Decryption successful. Embedding into cover video (This may take a while)...\n")
        self.root.update()

        # Run Embedding
        report, metrics = embed_video(self.cover_path, temp_decrypted_path, self.output_path, alpha=0.04)
        
        # Cleanup temporary decrypted file
        if os.path.exists(temp_decrypted_path):
            os.remove(temp_decrypted_path)

        self.result_text.insert(tk.END, report)
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()
